"""
Train MLP baselines over the predefined embedding experiment grid.

Run from repo root:
    python -m models.mlp.train
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from models.embedding_loader import experiment_dir_name, load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/mlp"

EXPERIMENTS = [
    ("one_hot", None, "one_hot"),
    ("one_hot", None, None),
    ("autoencoder/plain_ae", None, "cat_ae"),
    ("autoencoder/plain_ae", None, None),
    ("autoencoder/vae", None, "cat_ae"),
    ("autoencoder/vae", None, None),
    ("esm/cdr3", "autoencoder/plain_ae", "cat_ae"),
    ("esm/cdr3", "autoencoder/plain_ae", None),
    ("esm/full", "autoencoder/plain_ae", None),
]


ESM_PROJ_DIM = 128  # Project high-dim ESM embeddings down before the MLP layers


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        proj_dim: int | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        if proj_dim is not None:
            layers.extend([nn.Linear(prev_dim, proj_dim), nn.ReLU()])
            prev_dim = proj_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--embeddings_dir", default="outputs/embeddings")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--only_tcr", nargs="+", default=None,
                   help="Only run experiments whose TCR embedding matches one of these names.")
    p.add_argument("--force", action="store_true",
                   help="Re-run experiments whose row is already in results_summary.csv.")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    return p.parse_args()


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on the given set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1s)])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float(y_true.mean())
    return {
        "auroc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_experiment(
    tcr_embedding: str,
    peptide_embedding: str | None,
    categorical: str | None,
    args,
) -> dict | None:
    name = experiment_dir_name(tcr_embedding, peptide_embedding, categorical)
    logger.info("Running MLP experiment: %s", name)

    try:
        X_train, y_train = load_split(
            "train",
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            splits_dir=args.splits_dir,
            embeddings_dir=args.embeddings_dir,
            drop_missing=True,
        )
        X_val, y_val = load_split(
            "val",
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            splits_dir=args.splits_dir,
            embeddings_dir=args.embeddings_dir,
            drop_missing=True,
        )
        X_test, y_test = load_split(
            "test",
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            splits_dir=args.splits_dir,
            embeddings_dir=args.embeddings_dir,
            drop_missing=True,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning("Skipping experiment %s: %s", name, exc)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    proj_dim = ESM_PROJ_DIM if tcr_embedding.startswith("esm") else None
    model = MLPClassifier(input_dim=X_train.shape[1], dropout=args.dropout, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y_batch.size(0)
            n_train += y_batch.size(0)

        val_labels, val_probs = predict_proba(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probs)
        train_loss = running_loss / max(1, n_train)
        logger.info(
            "  epoch=%3d train_loss=%.4f val_auc=%.4f val_auprc=%.4f",
            epoch,
            train_loss,
            val_metrics["auroc"],
            val_metrics["auprc"],
        )

        if val_metrics["auroc"] > best_val_auc + 1e-4:
            best_val_auc = val_metrics["auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("Early stopping at epoch %d (best val AUROC=%.4f)", epoch, best_val_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs = predict_proba(model, val_loader, device)
    threshold = find_best_threshold(val_labels, val_probs)
    test_labels, test_probs = predict_proba(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs, threshold=threshold)

    experiment_out = os.path.join(args.out_dir, name)
    os.makedirs(experiment_out, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "config": {
                "input_dim": int(X_train.shape[1]),
                "hidden_dims": [256, 128, 64],
                "dropout": args.dropout,
                "proj_dim": proj_dim,
            },
        },
        os.path.join(experiment_out, "model.pt"),
    )
    save_json(
        os.path.join(experiment_out, "metrics.json"),
        {
            "experiment_name": name,
            "model_type": "mlp",
            "tcr_embedding": tcr_embedding,
            "peptide_embedding": peptide_embedding or tcr_embedding,
            "categorical": categorical,
            "input_dim": int(X_train.shape[1]),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "best_val_auroc": float(best_val_auc),
            "metrics": test_metrics,
        },
    )
    logger.info(
        "Completed %s | AUROC=%.4f AUPRC=%.4f ACC=%.4f F1=%.4f",
        name,
        test_metrics["auroc"],
        test_metrics["auprc"],
        test_metrics["accuracy"],
        test_metrics["f1"],
    )

    return {
        "experiment_name": name,
        "tcr_embedding": tcr_embedding,
        "peptide_embedding": peptide_embedding or tcr_embedding,
        "categorical": categorical or "none",
        "input_dim": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "best_val_auroc": float(best_val_auc),
        **test_metrics,
    }


def _already_in_summary(out_dir: str, name: str) -> bool:
    path = os.path.join(out_dir, "results_summary.csv")
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    return name in df["experiment_name"].astype(str).tolist()


def _update_results_summary(out_dir: str, new_rows: list[dict]) -> None:
    """Read existing results_summary.csv (if any), drop rows whose
    experiment_name matches a row in `new_rows`, append `new_rows`, write back."""
    if not new_rows:
        return
    path = os.path.join(out_dir, "results_summary.csv")
    new_df = pd.DataFrame(new_rows)
    new_names = set(new_df["experiment_name"])
    if os.path.exists(path) and os.path.getsize(path) > 0:
        old = pd.read_csv(path)
        old = old[~old["experiment_name"].isin(new_names)]
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(path, index=False)
    logger.info("Updated %s (%d new rows merged)", path, len(new_df))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only_tcr:
        wanted = set(args.only_tcr)
        experiments = [e for e in EXPERIMENTS if e[0] in wanted]
        logger.info("Filtering to TCR embeddings %s -> %d experiment(s)",
                    sorted(wanted), len(experiments))

    results: list[dict] = []
    for tcr_embedding, peptide_embedding, categorical in experiments:
        name = experiment_dir_name(tcr_embedding, peptide_embedding, categorical)
        if not args.force and _already_in_summary(args.out_dir, name):
            logger.info("Skipping %s: already in results_summary.csv (use --force to re-run)", name)
            continue
        result = run_experiment(
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            args=args,
        )
        if result is not None:
            results.append(result)
            _update_results_summary(args.out_dir, [result])

    logger.info("Done. %d new experiment row(s) merged.", len(results))


if __name__ == "__main__":
    main()
