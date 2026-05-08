"""
CNN TCR-peptide binding predictor over pre-computed embeddings.

Two parallel 1D-conv branches read the TCR and peptide embeddings, each is
globally pooled, and the pooled features (plus optional V/J + MHC metadata)
are passed through an MLP that outputs a binding logit.

Embeddings can be either:
  - per-residue (one_hot reshaped to (L, V)): in_channels = vocab dim,
    seq_len = padded sequence length.
  - vector (autoencoder / ESM pooled): in_channels = 1, seq_len = embed dim.

The model is agnostic to which it gets — the train script picks the right
shape based on the --embedding flag.
"""

import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    """Two Conv1D layers + global max-pool + linear projection."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        conv_channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.1,
        out_dim: int = 64,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(conv_channels, out_dim)
        self.seq_len = seq_len
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        h = self.conv(x)              # (B, conv_channels, L)
        h = self.pool(h).squeeze(-1)  # (B, conv_channels)
        return self.proj(h)           # (B, out_dim)


class EmbeddingCNN(nn.Module):
    """
    Dual-branch CNN over (TCR, peptide) embedding tensors.

    Args:
        tcr_in_channels: input channels for TCR conv (e.g. 22 for one-hot, 1 for vector AE/ESM)
        tcr_seq_len:     sequence length for TCR conv (e.g. 30 for one-hot, latent_dim for AE/ESM)
        pep_in_channels: input channels for peptide conv
        pep_seq_len:     sequence length for peptide conv
        conv_channels:   filters in each conv layer
        kernel_size:     conv kernel
        branch_out_dim:  size of each branch's pooled feature vector
        meta_dim:        size of optional metadata vector concatenated after pooling (0 to disable)
        dropout:         dropout in conv branches and classifier
    """

    def __init__(
        self,
        tcr_in_channels: int,
        tcr_seq_len: int,
        pep_in_channels: int,
        pep_seq_len: int,
        conv_channels: int = 64,
        kernel_size: int = 5,
        branch_out_dim: int = 64,
        meta_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.meta_dim = meta_dim

        self.tcr_branch = ConvBranch(
            tcr_in_channels, tcr_seq_len, conv_channels, kernel_size, dropout, branch_out_dim,
        )
        self.pep_branch = ConvBranch(
            pep_in_channels, pep_seq_len, conv_channels, kernel_size, dropout, branch_out_dim,
        )

        classifier_in = 2 * branch_out_dim + meta_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, branch_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_out_dim, 1),
        )

    def forward(
        self,
        tcr: torch.Tensor,            # (B, C_t, L_t)
        pep: torch.Tensor,            # (B, C_p, L_p)
        meta: torch.Tensor = None,    # (B, meta_dim) or None
    ) -> torch.Tensor:
        tcr_pool = self.tcr_branch(tcr)
        pep_pool = self.pep_branch(pep)
        feats = torch.cat([tcr_pool, pep_pool], dim=-1)
        if meta is not None and self.meta_dim > 0:
            feats = torch.cat([feats, meta], dim=-1)
        return self.classifier(feats).squeeze(-1)
