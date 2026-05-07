"""
Categorical feature augmentation via learned autoencoder latents (PepTCR-style).

Trains a small MLP autoencoder on the one-hot representation of each categorical
variable. The encoder bottleneck replaces the sparse one-hot vector, producing
dense low-dimensional representations that capture similarity between categories.

Architecture per variable:
    one-hot (vocab_size) → Linear → ReLU → latent (latent_dim)
                                               ↓
    one-hot (vocab_size) ← Linear ←────────── latent

After training, only the encoder half is used.

Reference: PepTCR-Net (Le et al. 2025) — "Categorical Autoencoding" of HLA
types and V/J gene segments.

Usage:
    from embeddings.feature_augment import CatAEFeatureAugmenter

    aug = CatAEFeatureAugmenter(latent_dim=32)
    aug.fit(train_df)                            # trains AEs + builds vocabs

    train_aug = aug.augment(train_embeddings, train_df)
    val_aug   = aug.augment(val_embeddings,   val_df)

    aug.save("outputs/feature_augment/cat_ae/")
    aug = CatAEFeatureAugmenter.load("outputs/feature_augment/cat_ae/")

CLI:
    python -m embeddings.feature_augment.autoencoder \\
        --embeddings outputs/embeddings/one_hot/combined_embeddings.npy \\
        --index      outputs/embeddings/one_hot/embedding_index.csv \\
        --data       processed/combined_trb_clean.csv \\
        --out        outputs/embeddings/one_hot/combined_augmented_cat_ae.npy \\
        --latent_dim 32
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from embeddings.feature_augment.one_hot import load_aligned_df, INDEX_COLS

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ("v_gene", "j_gene")


# ── Model ─────────────────────────────────────────────────────────────────────

class _CatAE(nn.Module):
    """Single-variable categorical autoencoder."""

    def __init__(self, vocab_size: int, latent_dim: int):
        super().__init__()
        # Wide hidden layer — aggressive compression from vocab_size needs capacity
        hidden = max(vocab_size // 2, latent_dim * 4, 128)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def _train_cat_ae(
    one_hot: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
) -> _CatAE:
    vocab_size = one_hot.shape[1]
    model = _CatAE(vocab_size, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # CrossEntropyLoss directly optimises for correct argmax reconstruction,
    # unlike BCEWithLogitsLoss which gets gamed by predicting all zeros.
    loss_fn = nn.CrossEntropyLoss()
    targets = torch.arange(vocab_size)  # class index for each one-hot row

    X = torch.tensor(one_hot, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, targets), batch_size=batch_size, shuffle=True)

    best_loss, wait = float("inf"), 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch, tgt in loader:
            batch, tgt = batch.to(device), tgt.to(device)
            loss = loss_fn(model(batch), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
            n += batch.size(0)
        epoch_loss = total / n
        if epoch_loss < best_loss - 1e-5:
            best_loss, wait = epoch_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"    Early stop at epoch {epoch} (loss={best_loss:.4f})")
                break
        if epoch % 10 == 0:
            logger.info(f"    epoch {epoch:3d}  loss={epoch_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu()


# ── Augmenter ─────────────────────────────────────────────────────────────────

class CatAEFeatureAugmenter:
    """
    Augments sequence embeddings with categorical autoencoder latents.

    Features:
      - mhc_class : 1-bit binary (unchanged — only 2 values, no compression needed)
      - v_gene    : latent_dim-dimensional learned encoding
      - j_gene    : latent_dim-dimensional learned encoding

    Total feature dim: 1 + 2 * latent_dim  (default: 65 with latent_dim=32).

    Args:
        latent_dim:  bottleneck size per categorical variable (default 32)
        epochs:      max training epochs per AE (default 100)
        batch_size:  training batch size (default 256)
        lr:          learning rate (default 1e-3)
        patience:    early stopping patience (default 10)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 10,
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        self._vocabs: dict[str, list] = {}
        self._models: dict[str, _CatAE] = {}
        self._fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CatAEFeatureAugmenter":
        """Build vocabularies and train one AE per categorical variable."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training categorical AEs on {device}")

        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            vocab = sorted(df[col].dropna().unique().tolist())
            self._vocabs[col] = vocab
            v2i = {v: i for i, v in enumerate(vocab)}

            # Build one-hot matrix over unique values (train on vocab, not all rows)
            n = len(vocab)
            one_hot = np.eye(n, dtype=np.float32)

            logger.info(f"  Training AE for {col} (vocab={n}, latent={self.latent_dim})...")
            self._models[col] = _train_cat_ae(
                one_hot, self.latent_dim, self.epochs,
                self.batch_size, self.lr, self.patience, device,
            )
            logger.info(f"  {col} done.")

        self._fitted = True
        logger.info(f"CatAEFeatureAugmenter fitted. feature_dim={self.feature_dim}")
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, feature_dim) float32 array of encoded metadata."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        parts = []

        # mhc_class: 1-bit binary (no compression needed)
        mhc = (df["mhc_class"].fillna("MHCI") == "MHCII").astype(np.float32).values
        parts.append(mhc.reshape(-1, 1))

        # v_gene, j_gene: encode via trained AE
        for col in CATEGORICAL_FEATURES:
            vocab = self._vocabs[col]
            v2i = {v: i for i, v in enumerate(vocab)}
            n = len(vocab)
            model = self._models[col]
            model.eval()

            # Build one-hot for each row (unknown/null → zero vector → zero latent)
            indices = df[col].map(v2i)
            oh = np.zeros((len(df), n), dtype=np.float32)
            valid = indices.notna()
            oh[valid.values, indices[valid].astype(int).values] = 1.0

            with torch.no_grad():
                latents = model.encode(torch.tensor(oh)).numpy()
            parts.append(latents)

        return np.concatenate(parts, axis=1)

    # ── Augment ───────────────────────────────────────────────────────────────

    def augment(self, embeddings: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Concatenate (N, d) embeddings with (N, feature_dim) metadata → (N, d+feature_dim)."""
        if len(embeddings) != len(df):
            raise ValueError(f"embeddings has {len(embeddings)} rows but df has {len(df)} rows")
        meta = self.transform(df)
        out = np.concatenate([embeddings, meta], axis=1)
        logger.info(f"Augmented: {embeddings.shape} + {meta.shape} → {out.shape}")
        return out

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return 1 + self.latent_dim * len(CATEGORICAL_FEATURES)

    def feature_breakdown(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return {"mhc_class": 1, **{c: self.latent_dim for c in CATEGORICAL_FEATURES}}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocabs.pkl"), "wb") as f:
            pickle.dump(self._vocabs, f)
        config = {"latent_dim": self.latent_dim, "epochs": self.epochs,
                  "batch_size": self.batch_size, "lr": self.lr, "patience": self.patience}
        with open(os.path.join(directory, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        for col, model in self._models.items():
            torch.save(model.state_dict(), os.path.join(directory, f"{col}_ae.pt"))
        logger.info(f"Saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "CatAEFeatureAugmenter":
        with open(os.path.join(directory, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        aug = cls(**config)
        with open(os.path.join(directory, "vocabs.pkl"), "rb") as f:
            aug._vocabs = pickle.load(f)
        for col in CATEGORICAL_FEATURES:
            vocab_size = len(aug._vocabs[col])
            model = _CatAE(vocab_size, aug.latent_dim)
            model.load_state_dict(torch.load(os.path.join(directory, f"{col}_ae.pt"), weights_only=True))
            model.eval()
            aug._models[col] = model
        aug._fitted = True
        logger.info(f"Loaded from {directory}/ (feature_dim={aug.feature_dim})")
        return aug


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Augment embeddings with categorical AE latents.")
    p.add_argument("--embeddings",    required=True)
    p.add_argument("--index",         required=True)
    p.add_argument("--data",          default="processed/combined_trb_clean.csv")
    p.add_argument("--out",           required=True)
    p.add_argument("--augmenter_out", default=None,
                   help="Directory to save fitted augmenter for reuse")
    p.add_argument("--latent_dim",    type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--patience",      type=int,   default=10)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    embeddings, df = load_aligned_df(args.embeddings, args.index, args.data)

    aug = CatAEFeatureAugmenter(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    aug.fit(df)
    logger.info("Feature breakdown: " + str(aug.feature_breakdown()))

    augmented = aug.augment(embeddings, df)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, augmented)

    index_path = args.out.replace(".npy", "_index.csv")
    df[[c for c in INDEX_COLS if c in df.columns]].to_csv(index_path, index=True)
    logger.info(f"Saved to {args.out}  index → {index_path}")

    if args.augmenter_out:
        aug.save(args.augmenter_out)


if __name__ == "__main__":
    main()
