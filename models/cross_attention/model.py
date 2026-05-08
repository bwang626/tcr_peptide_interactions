"""
Cross-attention TCR-peptide binding predictor.

Each CDR3-peptide pair is encoded independently, then each sequence
cross-attends to the other so every TCR residue can gather context from
every peptide residue (and vice versa). The resulting context-enriched
representations are mean-pooled and fed to an MLP classifier.

This directly implements the "pairwise matrix / cross-attention" approach:
the attention weight matrix at each layer is exactly the (L_tcr x L_pep)
interaction score between every residue pair.

Optionally appends V/J gene and MHC class as metadata features after pooling
(same encoding as OneHotFeatureAugmenter).

Two input modes
---------------
Token mode (default, esm_dim=0):
    Inputs are integer token indices (B, L). An internal nn.Embedding maps
    them to (B, L, d_model).

ESM mode (tcr_feature_dim > 0):
    Inputs are pre-computed per-residue ESM hidden states (B, L, tcr_feature_dim).
    A learned linear projection maps them to (B, L, d_model).
    Use encode_esm() / forward_esm() instead of encode() / forward().
    Masks are float tensors (1=real residue, 0=padding) of shape (B, L).

Architecture per forward pass:
    [token embed | ESM projection] + positional encode
        ↓
    N x CrossAttentionBlock
        [ TCR self-attn → TCR cross-attends peptide → FF ]
        [ Pep self-attn → Pep cross-attends TCR    → FF ]
        ↓
    masked mean-pool TCR, masked mean-pool peptide
        ↓
    concat(tcr_pool, pep_pool, [metadata])
        ↓
    MLP → binding logit

Usage (token mode):
    model = CrossAttentionTCRPep(d_model=64, n_heads=4, n_layers=2)
    logits = model(tcr_idx, pep_idx, tcr_mask, pep_mask)

Usage (ESM mode):
    model = CrossAttentionTCRPep(d_model=256, n_heads=8, n_layers=2, tcr_feature_dim=1152)
    # tcr_esm: (B, L_cdr3, 1152)  pep_esm: (B, L_pep, 1152)
    logits = model.forward_esm(tcr_esm, pep_esm, tcr_mask, pep_mask)
"""

import math

import numpy as np
import torch
import torch.nn as nn

from embeddings.one_hot.model import AA_TO_IDX, VOCAB_SIZE, DEFAULT_MAX_LEN, PAD_TOKEN

TCR_MAX_LEN = DEFAULT_MAX_LEN["tcr"]      # 30
PEP_MAX_LEN = DEFAULT_MAX_LEN["peptide"]  # 15


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_sequence(seq: str, max_len: int) -> tuple[list[int], list[float]]:
    """Return (token_ids, mask) for a single sequence, padded to max_len."""
    pad = AA_TO_IDX[PAD_TOKEN]
    unk = AA_TO_IDX.get("X", pad)
    ids  = [AA_TO_IDX.get(a, unk) for a in seq[:max_len]]
    mask = [1.0] * len(ids) + [0.0] * (max_len - len(ids))
    ids  = ids + [pad] * (max_len - len(ids))
    return ids, mask


def collate_sequences(seqs: list[str], max_len: int, device: torch.device):
    """Batch-encode a list of sequences → (idx, mask) tensors on device."""
    ids_list, mask_list = zip(*[encode_sequence(s, max_len) for s in seqs])
    idx  = torch.tensor(ids_list,  dtype=torch.long,  device=device)
    mask = torch.tensor(mask_list, dtype=torch.float32, device=device)
    return idx, mask


# ── positional encoding ───────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ── cross-attention block ─────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    One layer of bidirectional cross-attention.

    Order per sequence:
      1. Self-attention (each sequence attends to itself)
      2. Cross-attention (each sequence attends to the other)
      3. Feed-forward

    Both sequences share the same weight matrices — parameter-efficient and
    forces a symmetric inductive bias.
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.norm_self  = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ff    = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def _key_padding_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert float mask (1=real, 0=pad) → bool key_padding_mask (True=ignore)."""
        return mask == 0

    def forward(
        self,
        x: torch.Tensor,        # (B, L_x, D)  — sequence being updated
        ctx: torch.Tensor,      # (B, L_ctx, D) — sequence providing context
        x_mask: torch.Tensor,   # (B, L_x)   float, 1=real
        ctx_mask: torch.Tensor, # (B, L_ctx) float, 1=real
    ) -> torch.Tensor:
        kpm_x   = self._key_padding_mask(x_mask)
        kpm_ctx = self._key_padding_mask(ctx_mask)

        # Self-attention
        sa, _ = self.self_attn(x, x, x, key_padding_mask=kpm_x)
        x = self.norm_self(x + self.drop(sa))

        # Cross-attention: x queries, ctx provides keys/values
        ca, _ = self.cross_attn(x, ctx, ctx, key_padding_mask=kpm_ctx)
        x = self.norm_cross(x + self.drop(ca))

        # Feed-forward
        x = self.norm_ff(x + self.drop(self.ff(x)))
        return x


# ── main model ────────────────────────────────────────────────────────────────

class CrossAttentionTCRPep(nn.Module):
    """
    Cross-attention binding predictor.

    Args:
        d_model:    residue embedding dimension
        n_heads:    attention heads (must divide d_model)
        n_layers:   number of CrossAttentionBlocks
        ff_dim:     feed-forward hidden dim (default 4 * d_model)
        dropout:    dropout rate
        meta_dim:   dimension of optional metadata vector appended after pooling
                    (set to 0 to disable; use OneHotFeatureAugmenter.feature_dim)
    """

    def __init__(
        self,
        d_model:  int   = 64,
        n_heads:  int   = 4,
        n_layers: int   = 2,
        ff_dim:   int   = None,
        dropout:  float = 0.1,
        meta_dim: int   = 0,
        tcr_feature_dim: int = 0,
    ):
        """
        Args:
            tcr_feature_dim: If > 0, the TCR side consumes pre-computed per-residue
                             features (e.g. ESM hidden states) instead of token IDs.
                             A Linear(tcr_feature_dim, d_model) projection is applied.
                             Use forward_esm() when this is set.
        """
        super().__init__()
        ff_dim = ff_dim or 4 * d_model

        self.d_model         = d_model
        self.meta_dim        = meta_dim
        self.tcr_feature_dim = tcr_feature_dim

        # Token mode: learned AA embedding (used when tcr_feature_dim == 0)
        self.aa_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=AA_TO_IDX[PAD_TOKEN])
        # When tcr_feature_dim > 0, project pre-computed per-residue features → d_model
        self.tcr_feature_proj = (nn.Linear(tcr_feature_dim, d_model)
                                 if tcr_feature_dim > 0 else None)
        self.pe = SinusoidalPE(d_model, max_len=max(TCR_MAX_LEN, PEP_MAX_LEN) + 4)

        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        classifier_in = 2 * d_model + meta_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool x (B, L, D) over real (non-pad) positions."""
        mask = mask.unsqueeze(-1)           # (B, L, 1)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def encode(
        self,
        tcr_input: torch.Tensor,  # (B, L_t) long token ids OR (B, L_t, F) float features
        pep_idx:  torch.Tensor,   # (B, L_p)
        tcr_mask: torch.Tensor,   # (B, L_t) float
        pep_mask: torch.Tensor,   # (B, L_p) float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (tcr_pool, pep_pool) each (B, d_model)."""
        if self.tcr_feature_proj is not None:
            tcr = self.pe(self.tcr_feature_proj(tcr_input))  # (B, L_t, D)
        else:
            tcr = self.pe(self.aa_embed(tcr_input))            # (B, L_t, D)
        pep = self.pe(self.aa_embed(pep_idx))                  # (B, L_p, D)

        for layer in self.layers:
            # Update TCR using peptide as context, then peptide using updated TCR
            tcr = layer(tcr, pep, tcr_mask, pep_mask)
            pep = layer(pep, tcr, pep_mask, tcr_mask)

        return self._masked_mean(tcr, tcr_mask), self._masked_mean(pep, pep_mask)

    def forward(
        self,
        tcr_input: torch.Tensor,
        pep_idx:  torch.Tensor,
        tcr_mask: torch.Tensor,
        pep_mask: torch.Tensor,
        meta:     torch.Tensor = None,  # (B, meta_dim) optional
    ) -> torch.Tensor:
        """Return raw logits (B,)."""
        tcr_pool, pep_pool = self.encode(tcr_input, pep_idx, tcr_mask, pep_mask)
        pair = torch.cat([tcr_pool, pep_pool], dim=-1)  # (B, 2D)
        if meta is not None and self.meta_dim > 0:
            pair = torch.cat([pair, meta], dim=-1)
        return self.classifier(pair).squeeze(-1)

    def encode_esm(
        self,
        tcr_esm:  torch.Tensor,   # (B, L_cdr3, esm_dim) — per-residue CDR3 embeddings
        pep_esm:  torch.Tensor,   # (B, L_pep,  esm_dim) — per-residue peptide embeddings
        tcr_mask: torch.Tensor,   # (B, L_cdr3) float, 1=real residue
        pep_mask: torch.Tensor,   # (B, L_pep)  float, 1=real residue
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ESM input path: project per-residue embeddings, then cross-attend.

        Returns (tcr_pool, pep_pool) each (B, d_model).
        Requires esm_dim > 0 at construction time.
        """
        if self.tcr_feature_proj is None:
            raise RuntimeError(
                "encode_esm() requires tcr_feature_dim > 0. "
                "Re-initialise with CrossAttentionTCRPep(tcr_feature_dim=1152)."
            )
        # Project per-residue features → d_model, then add positional encoding
        tcr = self.pe(self.tcr_feature_proj(tcr_esm))  # (B, L_cdr3, d_model)
        pep = self.pe(self.tcr_feature_proj(pep_esm))  # (B, L_pep,  d_model)

        for layer in self.layers:
            tcr = layer(tcr, pep, tcr_mask, pep_mask)
            pep = layer(pep, tcr, pep_mask, tcr_mask)

        return self._masked_mean(tcr, tcr_mask), self._masked_mean(pep, pep_mask)

    def forward_esm(
        self,
        tcr_esm:  torch.Tensor,   # (B, L_cdr3, esm_dim)
        pep_esm:  torch.Tensor,   # (B, L_pep,  esm_dim)
        tcr_mask: torch.Tensor,   # (B, L_cdr3)
        pep_mask: torch.Tensor,   # (B, L_pep)
        meta:     torch.Tensor = None,
    ) -> torch.Tensor:
        """ESM input path forward pass. Returns raw logits (B,)."""
        tcr_pool, pep_pool = self.encode_esm(tcr_esm, pep_esm, tcr_mask, pep_mask)
        pair = torch.cat([tcr_pool, pep_pool], dim=-1)
        if meta is not None and self.meta_dim > 0:
            pair = torch.cat([pair, meta], dim=-1)
        return self.classifier(pair).squeeze(-1)

    def predict_proba(
        self,
        tcr_seqs: list[str],
        pep_seqs: list[str],
        meta:     torch.Tensor = None,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> np.ndarray:
        self.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(tcr_seqs), batch_size):
                t = tcr_seqs[i : i + batch_size]
                p = pep_seqs[i : i + batch_size]
                ti, tm = collate_sequences(t, TCR_MAX_LEN, device)
                pi, pm = collate_sequences(p, PEP_MAX_LEN, device)
                m = meta[i : i + batch_size].to(device) if meta is not None else None
                logits = self(ti, pi, tm, pm, m)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)
