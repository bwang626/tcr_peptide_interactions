"""
Graph-based embeddings for TCR-peptide interaction prediction.

Each TCR-peptide pair is represented as a single heterogeneous graph:
  - Nodes: amino-acid residues from the TCR (≤30) and peptide (≤15)
  - Edge type 0: sequential TCR backbone (i, i+1)
  - Edge type 1: sequential peptide backbone
  - Edge type 2: all-pairs bipartite TCR ↔ peptide (potential contacts)

A 3-layer relational graph-attention encoder produces per-node embeddings.
Mean-pooling over TCR nodes and peptide nodes yields two vectors that are
concatenated into the pair embedding. A small MLP head on top of this
embedding predicts binding (positive vs. shuffled-negative).

Self-contained — uses only torch / numpy / pandas, no PyG / DGL dependency.

Usage:
    from embeddings.graph.model import GraphEmbedder
    embedder = GraphEmbedder()
    embedder.fit(df)                    # df has columns: cdr3, peptide
    Z = embedder.transform(df)          # (N, 2 * latent_dim)
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one_hot.model import AA_TO_IDX, VOCAB_SIZE, DEFAULT_MAX_LEN, PAD_TOKEN

logger = logging.getLogger(__name__)

NUM_EDGE_TYPES = 3  # 0=tcr-backbone, 1=peptide-backbone, 2=tcr<->peptide


# ─────────────────────────────────────────────
# Node featurisation
# ─────────────────────────────────────────────

def _encode_residues(seq: str, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (aa_idx[max_len], mask[max_len]) — 0 for pad positions."""
    pad = AA_TO_IDX[PAD_TOKEN]
    idx = [AA_TO_IDX.get(a, AA_TO_IDX["X"]) for a in seq[:max_len]]
    mask = [1] * len(idx) + [0] * (max_len - len(idx))
    idx = idx + [pad] * (max_len - len(idx))
    return np.asarray(idx, dtype=np.int64), np.asarray(mask, dtype=np.float32)


def build_pair_tensors(
    tcr_seqs: List[str],
    pep_seqs: List[str],
    tcr_max_len: int,
    pep_max_len: int,
) -> dict:
    """
    Convert a batch of TCR / peptide pairs into dense tensors for the GNN.

    Node ordering: [TCR positions 0..tcr_max_len-1, peptide positions 0..pep_max_len-1].
    Adjacency is the same shape for every pair, so we build the masked adjacency
    once and reuse it for the whole batch (only the residue-mask varies).
    """
    n = len(tcr_seqs)
    tcr_idx = np.zeros((n, tcr_max_len), dtype=np.int64)
    pep_idx = np.zeros((n, pep_max_len), dtype=np.int64)
    tcr_mask = np.zeros((n, tcr_max_len), dtype=np.float32)
    pep_mask = np.zeros((n, pep_max_len), dtype=np.float32)
    for i, (t, p) in enumerate(zip(tcr_seqs, pep_seqs)):
        tcr_idx[i], tcr_mask[i] = _encode_residues(t, tcr_max_len)
        pep_idx[i], pep_mask[i] = _encode_residues(p, pep_max_len)
    return {
        "tcr_idx":  torch.from_numpy(tcr_idx),
        "pep_idx":  torch.from_numpy(pep_idx),
        "tcr_mask": torch.from_numpy(tcr_mask),
        "pep_mask": torch.from_numpy(pep_mask),
    }


def build_edge_type_matrix(tcr_len: int, pep_len: int) -> torch.Tensor:
    """
    (N, N) integer matrix of edge types where N = tcr_len + pep_len.
    -1 means "no edge" (used as a mask). Self-loops are not added — the
    GNN layer handles residual connections separately.
    """
    n = tcr_len + pep_len
    et = -torch.ones(n, n, dtype=torch.long)
    # TCR backbone (undirected, sequential)
    for i in range(tcr_len - 1):
        et[i, i + 1] = 0
        et[i + 1, i] = 0
    # Peptide backbone
    for i in range(pep_len - 1):
        et[tcr_len + i, tcr_len + i + 1] = 1
        et[tcr_len + i + 1, tcr_len + i] = 1
    # Bipartite cross edges
    et[:tcr_len, tcr_len:] = 2
    et[tcr_len:, :tcr_len] = 2
    return et


# ─────────────────────────────────────────────
# Relational GAT layer
# ─────────────────────────────────────────────

class RGATLayer(nn.Module):
    """
    Multi-head graph attention with one linear projection per edge type.

    Designed for dense batched graphs where every graph has the same node
    layout (padded TCR + padded peptide). Padding positions are masked out
    of attention via the residue mask.

    Input:
        x:           (B, N, F_in)
        edge_type:   (N, N) long, values in [0, NUM_EDGE_TYPES) or -1 for "no edge"
        node_mask:   (B, N) float, 1 for real residue, 0 for pad
    Output:
        (B, N, F_out)
    """
    def __init__(self, in_dim: int, out_dim: int, num_edge_types: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads      = num_heads
        self.head_dim       = out_dim // num_heads
        self.num_edge_types = num_edge_types

        # One projection per edge type (relational GNN style)
        self.W = nn.Parameter(torch.empty(num_edge_types, in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

        # Attention vectors per head
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_type: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # Project messages once per edge type, then select per (i, j) by edge_type.
        # x: (B, N, F_in) → per-type projection: (R, B, N, H, D)
        per_type = torch.einsum("bnf,rfo->rbno", x, self.W).view(self.num_edge_types, B, N, H, D)

        # Build per-edge feature m[i, j] = projection of x[j] under W[edge_type[i, j]].
        # edge_type is (N, N); flatten into (N*N,) and gather along the type axis.
        et = edge_type.clamp(min=0)  # invalid will be masked out below
        valid_edge = (edge_type >= 0).to(x.dtype)  # (N, N)

        # per_type[r, b, j, h, d] → m[b, i, j, h, d] where r = et[i, j]
        # Expand et to (N, N, 1, 1, 1) and gather j along the source axis (we want messages from j).
        # We index per_type with r=et[i,j], producing m[b, i, j, h, d] = per_type[et[i,j], b, j, h, d]
        et_idx = et.view(N * N)                                  # (N*N,)
        # per_type indexed: (N*N, B, N, H, D) — too big. Instead build directly:
        # m[i, j] depends on j and et[i, j]. Reshape to gather:
        # gather_index along type dim:
        # Want per_type[et_idx, :, j, :, :] but j varies with the second flat index.
        # Use advanced indexing.
        i_idx = torch.arange(N, device=x.device).repeat_interleave(N)  # 0,0,...,1,1,...
        j_idx = torch.arange(N, device=x.device).repeat(N)             # 0,1,...,N-1,0,1,...
        m = per_type[et_idx, :, j_idx, :, :]                           # (N*N, B, H, D)
        m = m.permute(1, 0, 2, 3).contiguous().view(B, N, N, H, D)     # (B, i, j, H, D)

        # Attention scores: e[b, i, j, h] = (a_src · h_i) + (a_dst · h_j) using projected i and j
        # Use the bipartite/self projection (edge type 2) as the destination projection — the
        # "i" side. To keep the layer simple and parameter-light, we re-use the same projected
        # m for both sides: dst features come from m[b, i, i] (self), src from m[b, i, j].
        h_dst = m[:, torch.arange(N), torch.arange(N), :, :]           # (B, N, H, D) — diagonal
        # Score = LeakyReLU( a_dst · h_dst[i] + a_src · h_src[i, j] )
        e_dst = (h_dst * self.a_dst).sum(-1)                            # (B, N, H)
        e_src = (m * self.a_src).sum(-1)                                # (B, N, N, H)
        scores = F.leaky_relu(e_dst.unsqueeze(2) + e_src, negative_slope=0.2)  # (B, N, N, H)

        # Mask: drop invalid edges, padded source nodes, and padded destination nodes.
        edge_mask = valid_edge.view(1, N, N, 1)                                 # (1, N, N, 1)
        src_mask  = node_mask.view(B, 1, N, 1)                                  # mask src j
        dst_mask  = node_mask.view(B, N, 1, 1)                                  # mask dst i
        full_mask = edge_mask * src_mask * dst_mask
        scores = scores.masked_fill(full_mask == 0, float("-inf"))

        # Softmax over source (j) for each destination (i). Some rows may be all -inf
        # (e.g. fully padded destination); zero them out post-softmax to avoid NaNs.
        attn = torch.softmax(scores, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Aggregate: out[b, i, h, d] = sum_j attn[b, i, j, h] * m[b, i, j, h, d]
        out = (attn.unsqueeze(-1) * m).sum(dim=2)                               # (B, N, H, D)
        out = out.reshape(B, N, H * D)

        out = self.norm(out + self.skip(x))
        return F.elu(out)


# ─────────────────────────────────────────────
# GNN encoder + predictor
# ─────────────────────────────────────────────

class TCRPeptideGNN(nn.Module):
    """
    GNN that maps a (TCR, peptide) pair to a fixed-size embedding and a
    binding-probability logit.

    Forward returns (logits, pair_embedding, tcr_emb, pep_emb).
    """
    def __init__(
        self,
        tcr_max_len: int = DEFAULT_MAX_LEN["tcr"],
        pep_max_len: int = DEFAULT_MAX_LEN["peptide"],
        aa_embed_dim: int = 32,
        hidden_dim: int   = 64,
        latent_dim: int   = 64,
        num_layers: int   = 3,
        num_heads: int    = 4,
        dropout: float    = 0.1,
    ):
        super().__init__()
        self.tcr_max_len = tcr_max_len
        self.pep_max_len = pep_max_len
        self.latent_dim  = latent_dim

        n_total = tcr_max_len + pep_max_len
        self.aa_embed   = nn.Embedding(VOCAB_SIZE, aa_embed_dim, padding_idx=AA_TO_IDX[PAD_TOKEN])
        self.pos_embed  = nn.Embedding(n_total, aa_embed_dim)
        self.type_embed = nn.Embedding(2, aa_embed_dim)       # 0=tcr, 1=peptide

        in_dim = aa_embed_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(RGATLayer(
                in_dim, hidden_dim, NUM_EDGE_TYPES,
                num_heads=num_heads, dropout=dropout,
            ))
            in_dim = hidden_dim

        self.tcr_proj = nn.Linear(hidden_dim, latent_dim)
        self.pep_proj = nn.Linear(hidden_dim, latent_dim)

        self.head = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

        # Edge structure is fixed once tcr/pep lengths are known.
        self.register_buffer("edge_type", build_edge_type_matrix(tcr_max_len, pep_max_len))
        node_type = torch.zeros(n_total, dtype=torch.long)
        node_type[tcr_max_len:] = 1
        self.register_buffer("node_type", node_type)
        self.register_buffer("position_ids", torch.arange(n_total))

    # ── forward pieces ────────────────────────────────────────────────────────

    def _initial_features(self, tcr_idx: torch.Tensor, pep_idx: torch.Tensor) -> torch.Tensor:
        # tcr_idx: (B, tcr_max), pep_idx: (B, pep_max) → (B, N, F)
        node_idx = torch.cat([tcr_idx, pep_idx], dim=1)   # (B, N)
        x = self.aa_embed(node_idx)
        x = x + self.pos_embed(self.position_ids).unsqueeze(0)
        x = x + self.type_embed(self.node_type).unsqueeze(0)
        return x

    def encode(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the GNN and return (pair_emb, tcr_emb, pep_emb)."""
        x = self._initial_features(batch["tcr_idx"], batch["pep_idx"])
        node_mask = torch.cat([batch["tcr_mask"], batch["pep_mask"]], dim=1)  # (B, N)
        for layer in self.layers:
            x = layer(x, self.edge_type, node_mask)

        tcr_x = x[:, :self.tcr_max_len]
        pep_x = x[:, self.tcr_max_len:]
        tcr_m = batch["tcr_mask"].unsqueeze(-1)
        pep_m = batch["pep_mask"].unsqueeze(-1)

        # Masked mean pool
        tcr_emb = (tcr_x * tcr_m).sum(1) / tcr_m.sum(1).clamp(min=1.0)
        pep_emb = (pep_x * pep_m).sum(1) / pep_m.sum(1).clamp(min=1.0)
        tcr_emb = self.tcr_proj(tcr_emb)
        pep_emb = self.pep_proj(pep_emb)

        pair_emb = torch.cat([tcr_emb, pep_emb], dim=-1)
        return pair_emb, tcr_emb, pep_emb

    def forward(self, batch: dict):
        pair_emb, tcr_emb, pep_emb = self.encode(batch)
        logits = self.head(pair_emb).squeeze(-1)
        return logits, pair_emb, tcr_emb, pep_emb


# ─────────────────────────────────────────────
# Dataset with negative sampling
# ─────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Wraps positive (TCR, peptide) pairs from the dataset and generates
    shuffled-peptide negatives on the fly.

    Each __getitem__ returns a (positive, negative) tuple of dicts.
    The training loop concatenates them into a 2*batch tensor.
    """
    def __init__(self, tcr_seqs: List[str], pep_seqs: List[str],
                 tcr_max_len: int, pep_max_len: int, neg_per_pos: int = 1, seed: int = 0):
        assert len(tcr_seqs) == len(pep_seqs)
        self.tcrs = list(tcr_seqs)
        self.peps = list(pep_seqs)
        self.tcr_max_len = tcr_max_len
        self.pep_max_len = pep_max_len
        self.neg_per_pos = neg_per_pos
        self.rng = np.random.default_rng(seed)
        # For fast negative sampling: track which peptides each TCR is paired with
        # (so we don't accidentally sample a real positive as a negative).
        self.tcr_to_peps: dict = {}
        for t, p in zip(self.tcrs, self.peps):
            self.tcr_to_peps.setdefault(t, set()).add(p)
        self.unique_peps = list(set(self.peps))

    def __len__(self):
        return len(self.tcrs)

    def _sample_negative_peptide(self, tcr: str) -> str:
        forbidden = self.tcr_to_peps[tcr]
        for _ in range(20):
            cand = self.unique_peps[self.rng.integers(len(self.unique_peps))]
            if cand not in forbidden:
                return cand
        return cand  # rare: all candidates exhausted; accept anyway

    def __getitem__(self, i: int):
        t = self.tcrs[i]
        p = self.peps[i]
        neg_p = self._sample_negative_peptide(t)
        return (t, p, neg_p)


def collate_pairs(batch, tcr_max_len: int, pep_max_len: int):
    pos_t = [b[0] for b in batch]
    pos_p = [b[1] for b in batch]
    neg_p = [b[2] for b in batch]
    tcrs   = pos_t + pos_t                    # same TCRs for negatives
    peps   = pos_p + neg_p
    labels = torch.cat([torch.ones(len(batch)), torch.zeros(len(batch))])
    tensors = build_pair_tensors(tcrs, peps, tcr_max_len, pep_max_len)
    tensors["labels"] = labels
    return tensors


# ─────────────────────────────────────────────
# High-level embedder (mirrors AutoencoderEmbedder API)
# ─────────────────────────────────────────────

class GraphEmbedder:
    """
    End-to-end TCR-peptide graph embedder.

    Trains a relational-GAT on positive pairs vs. shuffled-peptide negatives,
    then exposes:
        - .transform(df) → (N, 2 * latent_dim) pair embedding
        - .predict_proba(df) → (N,) binding probability

    Args mirror the autoencoder embedder where reasonable.
    """
    def __init__(
        self,
        tcr_max_len: int = DEFAULT_MAX_LEN["tcr"],
        pep_max_len: int = DEFAULT_MAX_LEN["peptide"],
        latent_dim: int  = 64,
        hidden_dim: int  = 64,
        num_layers: int  = 3,
        num_heads: int   = 4,
        dropout: float   = 0.1,
        neg_per_pos: int = 1,
    ):
        self.tcr_max_len = tcr_max_len
        self.pep_max_len = pep_max_len
        self.latent_dim  = latent_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.dropout     = dropout
        self.neg_per_pos = neg_per_pos
        self.embedding_dim = 2 * latent_dim

        self.model = TCRPeptideGNN(
            tcr_max_len=tcr_max_len, pep_max_len=pep_max_len,
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
        )
        self._is_fitted = False

    # ── train ────────────────────────────────────────────────────────────────

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val:   Optional[pd.DataFrame] = None,
        tcr_col:     str = "cdr3",
        peptide_col: str = "peptide",
        epochs:     int = 30,
        batch_size: int = 128,
        lr:         float = 1e-3,
        device:     Optional[str] = None,
        patience:   int = 5,
    ) -> "GraphEmbedder":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        ds = PairDataset(
            df_train[tcr_col].tolist(), df_train[peptide_col].tolist(),
            self.tcr_max_len, self.pep_max_len, neg_per_pos=self.neg_per_pos,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_pairs(b, self.tcr_max_len, self.pep_max_len),
        )

        val_loader = None
        if df_val is not None and len(df_val):
            ds_val = PairDataset(
                df_val[tcr_col].tolist(), df_val[peptide_col].tolist(),
                self.tcr_max_len, self.pep_max_len, neg_per_pos=self.neg_per_pos, seed=1,
            )
            val_loader = DataLoader(
                ds_val, batch_size=batch_size, shuffle=False,
                collate_fn=lambda b: collate_pairs(b, self.tcr_max_len, self.pep_max_len),
            )

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, patience // 2), factor=0.5)
        bce = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss, train_correct, train_n = 0.0, 0, 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, *_ = self.model(batch)
                loss = bce(logits, batch["labels"])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                train_loss    += loss.item() * batch["labels"].size(0)
                train_correct += ((logits > 0).float() == batch["labels"]).sum().item()
                train_n       += batch["labels"].size(0)

            train_loss /= train_n
            train_acc = train_correct / train_n

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, device)
                sched.step(val_loss)
                logger.info(
                    f"Epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.3f}"
                    f" | val loss={val_loss:.4f} acc={val_acc:.3f}"
                )
                if val_loss < best_val - 1e-4:
                    best_val   = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            else:
                logger.info(f"Epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.3f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        self._is_fitted = True
        return self

    @torch.no_grad()
    def _evaluate(self, loader, device) -> Tuple[float, float]:
        self.model.eval()
        bce = nn.BCEWithLogitsLoss(reduction="sum")
        total_loss, correct, n = 0.0, 0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, *_ = self.model(batch)
            total_loss += bce(logits, batch["labels"]).item()
            correct    += ((logits > 0).float() == batch["labels"]).sum().item()
            n          += batch["labels"].size(0)
        return total_loss / n, correct / n

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def transform(
        self, df: pd.DataFrame,
        tcr_col: str = "cdr3", peptide_col: str = "peptide",
        batch_size: int = 256,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .transform()")
        self.model.eval()
        tcr_seqs = df[tcr_col].tolist()
        pep_seqs = df[peptide_col].tolist()

        embs = []
        for i in range(0, len(tcr_seqs), batch_size):
            chunk = build_pair_tensors(
                tcr_seqs[i:i + batch_size], pep_seqs[i:i + batch_size],
                self.tcr_max_len, self.pep_max_len,
            )
            pair, _, _ = self.model.encode(chunk)
            embs.append(pair.cpu().numpy())
        return np.concatenate(embs, axis=0)

    @torch.no_grad()
    def predict_proba(
        self, df: pd.DataFrame,
        tcr_col: str = "cdr3", peptide_col: str = "peptide",
        batch_size: int = 256,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict_proba()")
        self.model.eval()
        tcr_seqs = df[tcr_col].tolist()
        pep_seqs = df[peptide_col].tolist()

        probs = []
        for i in range(0, len(tcr_seqs), batch_size):
            chunk = build_pair_tensors(
                tcr_seqs[i:i + batch_size], pep_seqs[i:i + batch_size],
                self.tcr_max_len, self.pep_max_len,
            )
            logits, *_ = self.model(chunk)
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs, axis=0)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": {
                "tcr_max_len": self.tcr_max_len,
                "pep_max_len": self.pep_max_len,
                "latent_dim":  self.latent_dim,
                "hidden_dim":  self.hidden_dim,
                "num_layers":  self.num_layers,
                "num_heads":   self.num_heads,
                "dropout":     self.dropout,
                "neg_per_pos": self.neg_per_pos,
            },
        }, path)
        logger.info(f"Saved graph embedder to {path}")

    @classmethod
    def load(cls, path: str) -> "GraphEmbedder":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        emb = cls(**cfg)
        emb.model.load_state_dict(ckpt["state_dict"])
        emb._is_fitted = True
        return emb
