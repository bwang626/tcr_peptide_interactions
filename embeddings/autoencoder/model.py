"""
Autoencoder-based embeddings for TCR and peptide sequences.

Encodes amino acid sequences (CDR3/TCRβ chains and peptide epitopes) into
fixed-length dense latent vectors using a sequence-to-latent autoencoder.

Architecture:
  - Input: one-hot encoded amino acid sequence (padded to max_len)
  - Encoder: Conv1D layers → GRU → dense latent vector (z)
  - Decoder: dense → GRU → Conv1D → reconstructed one-hot sequence
  - Training objective: cross-entropy reconstruction loss

Usage:
    from embeddings.autoencoder.model import SequenceAutoencoder, AutoencoderEmbedder

    ae = SequenceAutoencoder(seq_type="tcr")
    ae.fit(tcr_sequences)

    embedder = AutoencoderEmbedder(tcr_ae=ae, peptide_ae=peptide_ae)
    embeddings = embedder.transform(df)   # df has columns: cdr3, peptide
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple
import logging

# ── Import shared encoding utilities from one_hot ────────────────────────────
# one_hot/ is the single source of truth for AA encoding constants and helpers.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one_hot.model import (
    encode_sequence,
    encode_sequences,
    decode_one_hot,
    AA_TO_IDX,
    VOCAB_SIZE,
    DEFAULT_MAX_LEN,
    PAD_TOKEN,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """
    Convolutional + GRU encoder.
    Input:  (batch, seq_len, vocab_size)
    Output: (batch, latent_dim)
    """
    def __init__(self, seq_len: int, vocab_size: int, latent_dim: int,
                 conv_channels: int = 64, kernel_size: int = 3,
                 gru_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(vocab_size, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(conv_channels, gru_hidden, batch_first=True,
                          bidirectional=True, num_layers=1)
        self.fc_mu     = nn.Linear(gru_hidden * 2, latent_dim)
        self.fc_logvar = nn.Linear(gru_hidden * 2, latent_dim)  # VAE only; ignored in plain AE

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, V) → conv expects (B, V, L)
        h = self.conv(x.permute(0, 2, 1))              # (B, C, L)
        h = h.permute(0, 2, 1)                         # (B, L, C)
        _, hidden = self.gru(h)                        # hidden: (2, B, H)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # (B, 2H)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class ConvDecoder(nn.Module):
    """
    GRU + convolutional decoder.
    Input:  (batch, latent_dim)
    Output: (batch, seq_len, vocab_size) — raw logits
    """
    def __init__(self, seq_len: int, vocab_size: int, latent_dim: int,
                 conv_channels: int = 64, kernel_size: int = 3,
                 gru_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len    = seq_len
        self.gru_hidden = gru_hidden
        self.fc  = nn.Linear(latent_dim, gru_hidden * seq_len)
        self.gru = nn.GRU(gru_hidden, gru_hidden, batch_first=True, num_layers=2)
        self.conv = nn.Sequential(
            nn.Conv1d(gru_hidden, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, vocab_size, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.fc(z).view(B, self.seq_len, self.gru_hidden)  # (B, L, H)
        h, _ = self.gru(h)                                      # (B, L, H)
        return self.conv(h.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, V)


# ─────────────────────────────────────────────
# Main autoencoder (with optional VAE mode)
# ─────────────────────────────────────────────

class SequenceAutoencoder(nn.Module):
    """
    Sequence autoencoder for amino acid sequences.

    Supports two modes:
      - vae=False (default): plain autoencoder, z = mu
      - vae=True:            variational autoencoder, z ~ N(mu, sigma)
                             adds KL divergence term to loss (good regulariser,
                             makes latent space smoother for downstream models)

    Args:
        seq_type:      "tcr" or "peptide" — sets default max_len
        max_len:       override max sequence length
        latent_dim:    size of the embedding vector (default 64)
        conv_channels: convolutional filter count
        gru_hidden:    GRU hidden size
        dropout:       dropout rate
        vae:           use variational mode
        kl_weight:     weight of KL term if vae=True (beta-VAE style)
    """
    def __init__(
        self,
        seq_type:      str   = "tcr",
        max_len:       Optional[int] = None,
        latent_dim:    int   = 64,
        conv_channels: int   = 64,
        gru_hidden:    int   = 128,
        dropout:       float = 0.1,
        vae:           bool  = False,
        kl_weight:     float = 0.01,
    ):
        super().__init__()
        self.seq_type  = seq_type
        self.max_len   = max_len or DEFAULT_MAX_LEN.get(seq_type, 30)
        self.latent_dim = latent_dim
        self.vae       = vae
        self.kl_weight = kl_weight

        self.encoder = ConvEncoder(self.max_len, VOCAB_SIZE, latent_dim,
                                   conv_channels, gru_hidden=gru_hidden, dropout=dropout)
        self.decoder = ConvDecoder(self.max_len, VOCAB_SIZE, latent_dim,
                                   conv_channels, gru_hidden=gru_hidden, dropout=dropout)
        self._is_fitted = False

    # ── forward ──────────────────────────────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.vae and self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu  # deterministic at inference or in plain AE mode

    def forward(self, x: torch.Tensor):
        """x: (B, L, V) one-hot. Returns (logits, mu, logvar)."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns latent vector z (mu) for a batch. Shape: (B, latent_dim)."""
        mu, _ = self.encoder(x)
        return mu

    # ── loss ─────────────────────────────────────────────────────────────────

    def loss(self, x: torch.Tensor, logits: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss (cross-entropy over AA vocab) + optional KL term.
        x:      (B, L, V) one-hot targets
        logits: (B, L, V) decoder output
        """
        targets = x.argmax(dim=-1)  # (B, L)
        recon = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=AA_TO_IDX[PAD_TOKEN],  # don't penalise pad positions
        )
        if self.vae:
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon + self.kl_weight * kl
        return recon

    # ── training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        sequences:     List[str],
        epochs:        int = 50,
        batch_size:    int = 256,
        lr:            float = 1e-3,
        device:        Optional[str] = None,
        val_sequences: Optional[List[str]] = None,
        patience:      int = 10,
    ) -> "SequenceAutoencoder":
        """
        Train the autoencoder on a list of amino acid strings.

        Args:
            sequences:      training sequences
            epochs:         max training epochs
            batch_size:     mini-batch size
            lr:             Adam learning rate
            device:         "cpu", "cuda", or None (auto-detect)
            val_sequences:  optional validation set for early stopping
            patience:       early-stopping patience (epochs)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        X      = torch.tensor(encode_sequences(sequences, self.max_len))
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_sequences:
            Xv = torch.tensor(encode_sequences(val_sequences, self.max_len))
            val_loader = DataLoader(TensorDataset(Xv), batch_size=batch_size)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience // 2, factor=0.5)

        best_val_loss   = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.train()
            train_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits, mu, logvar = self(batch)
                loss = self.loss(batch, logits, mu, logvar)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * batch.size(0)
            train_loss /= len(sequences)

            if val_loader is not None:
                val_loss = self._evaluate(val_loader, device)
                scheduler.step(val_loss)
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    self._best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        self.load_state_dict(self._best_state)
                        break
            else:
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch:3d} | train={train_loss:.4f}")

        self._is_fitted = True
        self.to("cpu")
        return self

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, device: str) -> float:
        self.eval()
        total, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            logits, mu, logvar = self(batch)
            total += self.loss(batch, logits, mu, logvar).item() * batch.size(0)
            n += batch.size(0)
        return total / n

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def transform(self, sequences: List[str], batch_size: int = 512) -> np.ndarray:
        """
        Encode a list of sequences → latent vectors.
        Returns numpy array of shape (N, latent_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .transform()")
        self.eval()
        X      = torch.tensor(encode_sequences(sequences, self.max_len))
        loader = DataLoader(TensorDataset(X), batch_size=batch_size)
        zs = []
        for (batch,) in loader:
            zs.append(self.encode(batch).numpy())
        return np.concatenate(zs, axis=0)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "seq_type":  self.seq_type,
                "max_len":   self.max_len,
                "latent_dim": self.latent_dim,
                "vae":       self.vae,
                "kl_weight": self.kl_weight,
            }
        }, path)
        logger.info(f"Saved autoencoder to {path}")

    @classmethod
    def load(cls, path: str) -> "SequenceAutoencoder":
        checkpoint = torch.load(path, map_location="cpu")
        cfg   = checkpoint["config"]
        model = cls(**cfg)
        model.load_state_dict(checkpoint["state_dict"])
        model._is_fitted = True
        return model


# ─────────────────────────────────────────────
# Joint embedder: combines TCR + peptide AEs
# ─────────────────────────────────────────────

class AutoencoderEmbedder:
    """
    Wraps two fitted SequenceAutoencoders (one for TCRβ CDR3, one for peptide)
    and produces a combined embedding for each TCR-peptide pair.

    Args:
        tcr_ae:     fitted SequenceAutoencoder for CDR3β
        peptide_ae: fitted SequenceAutoencoder for peptide epitope
        concat:     if True (default), concatenate the two latent vectors.
                    if False, return them separately as a dict.
    """
    def __init__(self, tcr_ae: SequenceAutoencoder,
                 peptide_ae: SequenceAutoencoder,
                 concat: bool = True):
        self.tcr_ae     = tcr_ae
        self.peptide_ae = peptide_ae
        self.concat     = concat
        self.embedding_dim = tcr_ae.latent_dim + peptide_ae.latent_dim

    def transform(self, df, tcr_col: str = "cdr3", peptide_col: str = "peptide"):
        """
        Embed a DataFrame of TCR-peptide pairs.

        Args:
            df:          pandas DataFrame with TCR and peptide columns
            tcr_col:     column name for CDR3β sequences (default: "cdr3")
            peptide_col: column name for peptide epitope sequences

        Returns:
            if concat=True:  numpy array of shape (N, tcr_latent_dim + peptide_latent_dim)
            if concat=False: dict with keys "tcr" and "peptide"
        """
        tcr_z = self.tcr_ae.transform(df[tcr_col].tolist())
        pep_z = self.peptide_ae.transform(df[peptide_col].tolist())
        if self.concat:
            return np.concatenate([tcr_z, pep_z], axis=1)
        return {"tcr": tcr_z, "peptide": pep_z}