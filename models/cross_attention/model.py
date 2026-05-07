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

Architecture per forward pass:
    AA embed + positional encode
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

Usage:
    model = CrossAttentionTCRPep(d_model=64, n_heads=4, n_layers=2)
    logits = model(tcr_idx, pep_idx, tcr_mask, pep_mask)          # no metadata
    logits = model(tcr_idx, pep_idx, tcr_mask, pep_mask, meta)    # with metadata
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
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * d_model

        self.d_model  = d_model
        self.meta_dim = meta_dim

        self.aa_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=AA_TO_IDX[PAD_TOKEN])
        self.pe       = SinusoidalPE(d_model, max_len=max(TCR_MAX_LEN, PEP_MAX_LEN) + 4)

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
        tcr_idx:  torch.Tensor,   # (B, L_t)
        pep_idx:  torch.Tensor,   # (B, L_p)
        tcr_mask: torch.Tensor,   # (B, L_t) float
        pep_mask: torch.Tensor,   # (B, L_p) float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (tcr_pool, pep_pool) each (B, d_model)."""
        tcr = self.pe(self.aa_embed(tcr_idx))  # (B, L_t, D)
        pep = self.pe(self.aa_embed(pep_idx))  # (B, L_p, D)

        for layer in self.layers:
            # Update TCR using peptide as context, then peptide using updated TCR
            tcr = layer(tcr, pep, tcr_mask, pep_mask)
            pep = layer(pep, tcr, pep_mask, tcr_mask)

        return self._masked_mean(tcr, tcr_mask), self._masked_mean(pep, pep_mask)

    def forward(
        self,
        tcr_idx:  torch.Tensor,
        pep_idx:  torch.Tensor,
        tcr_mask: torch.Tensor,
        pep_mask: torch.Tensor,
        meta:     torch.Tensor = None,  # (B, meta_dim) optional
    ) -> torch.Tensor:
        """Return raw logits (B,)."""
        tcr_pool, pep_pool = self.encode(tcr_idx, pep_idx, tcr_mask, pep_mask)
        pair = torch.cat([tcr_pool, pep_pool], dim=-1)  # (B, 2D)
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
