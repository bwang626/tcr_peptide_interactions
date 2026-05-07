"""
eval_layers.py

Identify which ESMplusplus transformer layer produces the most linearly
separable TCR embeddings for epitope classification.

Task
----
Multi-class prediction of which peptide a TCR binds, using only the
mean-pooled CDR3 (or full-sequence) embedding extracted from each
candidate layer. A Logistic Regression is used as a fast linear probe —
the goal is to measure raw representational separability, not to build a
strong final classifier.

Key efficiency note
-------------------
All candidate layers are extracted from a SINGLE forward pass per batch
(output_hidden_states=True). Compute cost is therefore independent of
how many layers are tested.

Run from the repo root:
    python embeddings/esm/eval_layers.py
    python embeddings/esm/eval_layers.py --layers 12 24 30 35 36
    python embeddings/esm/eval_layers.py --n_peptides 3 --samples 300
    python embeddings/esm/eval_layers.py --region full

Outputs (outputs/embeddings/esm/):
    layer_eval_results.csv    per-layer accuracy and macro F1 (mean ± std)
    layer_eval_f1.png         bar chart comparing layers
"""

import argparse
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils._config_module as _cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL   = "Synthyra/ESMplusplus_large"
DEFAULT_LAYERS  = [12, 24, 30, 35, 36]
DEFAULT_N_PEP   = 5
DEFAULT_SAMPLES = 500
DEFAULT_N_FOLDS = 5
BASE_OUT        = "outputs/embeddings/esm"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_name: str, device: str):
    orig = _cm.ConfigModule.__setattr__

    def _lenient(self, name, value):
        try:
            orig(self, name, value)
        except AttributeError:
            object.__setattr__(self, name, value)

    _cm.ConfigModule.__setattr__ = _lenient
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    _cm.ConfigModule.__setattr__ = orig
    return model.to(device).eval()


def _detect_bos_offset(tokenizer) -> int:
    probe = "ACDEFGHIKLM"
    ids = tokenizer([probe], return_tensors="pt")["input_ids"][0]
    return len(ids) - len(probe) - (1 if len(ids) > len(probe) + 1 else 0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_balanced_subset(
    df: pd.DataFrame,
    n_peptides: int,
    samples_per_peptide: int,
    seed: int,
) -> pd.DataFrame:
    """Keep top-N peptides by count and draw a balanced sample from each."""
    top = df["peptide"].value_counts().head(n_peptides).index.tolist()
    logger.info(f"Top {n_peptides} peptides: {top}")

    parts = []
    for pep in top:
        pool = df[df["peptide"] == pep]
        n = min(samples_per_peptide, len(pool))
        if n < samples_per_peptide:
            logger.warning(f"  {pep}: only {len(pool)} rows available, using all {n}")
        parts.append(pool.sample(n=n, random_state=seed))

    subset = pd.concat(parts, ignore_index=True)
    counts = subset["peptide"].value_counts()
    logger.info(f"Balanced subset: {len(subset)} rows across {len(top)} classes")
    for pep, cnt in counts.items():
        logger.info(f"  {pep}: {cnt}")
    return subset


# ---------------------------------------------------------------------------
# Embedding extraction — all layers in one forward pass per batch
# ---------------------------------------------------------------------------

def extract_all_layers(
    model,
    tokenizer,
    df: pd.DataFrame,
    target_layers: list[int],
    region: str,
    device: str,
    bos_offset: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Return mean-pooled embeddings for every target layer.

    Extracts all requested layers from a single forward pass per batch.

    Parameters
    ----------
    region : "cdr3" pools over CDR3 residues only (cdr3_start:cdr3_end).
             "full" pools over the entire mature sequence.

    Returns
    -------
    dict mapping layer_idx → float32 array (N, D).
    Rows where the required span is unavailable are filled with NaN.
    """
    N = len(df)
    hidden_dim = model.embed.embedding_dim
    buffers: dict[int, np.ndarray] = {
        l: np.full((N, hidden_dim), np.nan, dtype=np.float32)
        for l in target_layers
    }

    # Length-sorted order keeps padding minimal
    order = df["full_aa"].str.len().argsort().tolist()
    n_batches = (len(order) + batch_size - 1) // batch_size

    t_start = time.perf_counter()

    for b_idx, start in enumerate(range(0, len(order), batch_size)):
        batch_rows = order[start : start + batch_size]
        seqs = [df.at[i, "full_aa"] for i in batch_rows]
        spans = [
            (
                df.at[i, "cdr3_start"] if pd.notna(df.at[i, "cdr3_start"]) else None,
                df.at[i, "cdr3_end"]   if pd.notna(df.at[i, "cdr3_end"])   else None,
            )
            for i in batch_rows
        ]

        enc = tokenizer(seqs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # Pull the required layers; pool each immediately to avoid keeping
        # all full (B, L, D) tensors alive simultaneously.
        for layer_idx in target_layers:
            hidden = out.hidden_states[layer_idx]  # (B, L_padded, D)

            for local_i, (global_i, seq, (cs, ce)) in enumerate(
                zip(batch_rows, seqs, spans)
            ):
                tok_start = bos_offset
                tok_end   = bos_offset + len(seq)
                residues  = hidden[local_i, tok_start:tok_end, :]  # (seq_len, D)

                if region == "cdr3":
                    if cs is None or ce is None:
                        continue
                    residues = residues[int(cs) : int(ce), :]

                vec = residues.mean(dim=0)
                buffers[layer_idx][global_i] = vec.cpu().float().numpy()

        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
            elapsed = time.perf_counter() - t_start
            n_done  = min(start + batch_size, N)
            rate    = n_done / elapsed
            eta     = (N - n_done) / rate if rate > 0 else 0.0
            logger.info(
                f"  Batch {b_idx + 1}/{n_batches}  "
                f"({n_done}/{N} seqs)  "
                f"{elapsed:.1f}s elapsed  "
                f"~{eta:.0f}s remaining  "
                f"({rate:.1f} seqs/s)"
            )

    total = time.perf_counter() - t_start
    logger.info(
        f"Embedding extraction complete: {total:.1f}s total  "
        f"({N / total:.1f} seqs/s)"
    )
    return buffers


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_path(out_dir: str) -> str:
    return os.path.join(out_dir, "layer_eval_cache.npz")


def _save_cache(
    path: str,
    buffers: dict[int, np.ndarray],
    y: np.ndarray,
    classes: list,
    layers: list[int],
    region: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {f"layer_{l}": buffers[l] for l in layers}
    payload["y"]       = y
    payload["classes"] = np.array(classes, dtype=object)
    payload["layers"]  = np.array(layers)
    payload["region"]  = np.array(region)
    np.savez_compressed(path, **payload)
    logger.info(f"Embedding cache saved to {path}")


def _load_cache(
    path: str,
    layers: list[int],
    region: str,
) -> tuple[dict[int, np.ndarray], np.ndarray, list] | None:
    """Return (buffers, y, classes) from cache, or None if cache is missing or stale."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    if sorted(data["layers"].tolist()) != sorted(layers):
        logger.info("Cache layers don't match requested layers — re-extracting.")
        return None
    if str(data["region"]) != region:
        logger.info("Cache region doesn't match --region — re-extracting.")
        return None
    buffers = {l: data[f"layer_{l}"] for l in layers}
    return buffers, np.asarray(data["y"]), data["classes"].tolist()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def evaluate_layer(X: np.ndarray, y: np.ndarray, n_folds: int) -> dict[str, float]:
    """Stratified K-fold CV with a standardised Logistic Regression linear probe.

    StandardScaler is fitted inside each fold to prevent data leakage.
    Returns mean and std of accuracy and macro F1 across folds.
    """
    clf = make_pipeline(
        StandardScaler(),
        # multi_class removed in scikit-learn 1.6; lbfgs handles multinomial automatically
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1),
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1,
    )
    return {
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std":  float(scores["test_accuracy"].std()),
        "f1_mean":       float(scores["test_f1_macro"].mean()),
        "f1_std":        float(scores["test_f1_macro"].std()),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(
    results: list[dict],
    out_path: str,
    n_folds: int,
    n_classes: int,
    num_blocks: int,
    region: str,
) -> None:
    layers    = [r["layer"] for r in results]
    f1_means  = np.array([r["f1_mean"] for r in results])
    f1_stds   = np.array([r["f1_std"] for r in results])
    acc_means = np.array([r["accuracy_mean"] for r in results])
    acc_stds  = np.array([r["accuracy_std"] for r in results])

    x     = np.arange(len(layers))
    width = 0.35
    baseline = 1.0 / n_classes

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 1.8), 5))

    bars_f1 = ax.bar(
        x - width / 2, f1_means, width,
        yerr=f1_stds, capsize=5,
        label="Macro F1", color="#4C72B0",
        error_kw={"elinewidth": 1.4, "ecolor": "#2c4e82"},
    )
    ax.bar(
        x + width / 2, acc_means, width,
        yerr=acc_stds, capsize=5,
        label="Accuracy", color="#DD8452", alpha=0.9,
        error_kw={"elinewidth": 1.4, "ecolor": "#a0522d"},
    )

    # Random-chance baseline
    ax.axhline(
        baseline, color="grey", linestyle="--", linewidth=1.2,
        label=f"Random baseline ({baseline:.2f})",
    )

    # Highlight the best F1 bar
    best_idx = int(np.argmax(f1_means))
    bars_f1[best_idx].set_edgecolor("gold")
    bars_f1[best_idx].set_linewidth(2.8)

    # Annotate F1 values above each bar
    for bar, mean, std in zip(bars_f1, f1_means, f1_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.012,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=8, color="#2c4e82", fontweight="bold",
        )

    # X-axis labels — annotate special layers
    x_labels = []
    for l in layers:
        if l == num_blocks:
            x_labels.append(f"Layer {l}\n(final, normed)")
        elif l == num_blocks - 1:
            x_labels.append(f"Layer {l}\n(2nd-to-last)")
        else:
            x_labels.append(f"Layer {l}")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)

    region_label = "CDR3 region" if region == "cdr3" else "full sequence"
    ax.set_ylabel("Score")
    ax.set_title(
        f"ESMplusplus Layer Evaluation — Epitope Classification\n"
        f"Mean-pooled {region_label} embedding · {n_classes}-class · "
        f"{n_folds}-fold CV · Logistic Regression",
        fontsize=10,
    )
    ax.set_ylim(0, min(1.0, max(acc_means.max(), f1_means.max()) + 0.15))
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",       default="processed/esm_trb_dataset.csv")
    p.add_argument("--out",        default=BASE_OUT)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument(
        "--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
        help="1-indexed transformer layers to evaluate (default: 12 24 30 35 36)",
    )
    p.add_argument(
        "--region", choices=["cdr3", "full"], default="cdr3",
        help="Sequence region to pool over: 'cdr3' (default) or 'full' sequence",
    )
    p.add_argument(
        "--n_peptides", type=int, default=DEFAULT_N_PEP,
        help="Number of top peptides to use as classes (default: 5)",
    )
    p.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help="Rows sampled per peptide class (default: 500)",
    )
    p.add_argument("--n_folds",    type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument(
        "--force_extraction", action="store_true",
        help="Ignore any cached embeddings and re-run the forward passes.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cpu":
        logger.warning(
            "Running on CPU — forward passes will be slow. "
            "Consider using --samples 100 for a quick test."
        )

    logger.info(f"Loading {args.model} ...")
    model      = _load_model(args.model, device)
    tokenizer  = model.tokenizer
    num_blocks = len(model.transformer.blocks)
    bos_offset = _detect_bos_offset(tokenizer)
    logger.info(f"  {num_blocks} blocks, hidden_dim={model.embed.embedding_dim}, BOS offset={bos_offset}")

    # Validate requested layers against model depth
    bad = [l for l in args.layers if not 1 <= l <= num_blocks]
    if bad:
        raise ValueError(f"Layers out of range 1..{num_blocks}: {bad}")
    layers_sorted = sorted(set(args.layers))

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False)

    required = ["full_aa"]
    if args.region == "cdr3":
        required += ["cdr3_start", "cdr3_end"]
    df = df.dropna(subset=required).reset_index(drop=True)
    logger.info(f"  {len(df):,} usable rows")

    subset = build_balanced_subset(df, args.n_peptides, args.samples, args.seed)

    le = LabelEncoder()
    y  = le.fit_transform(subset["peptide"].values)
    n_classes = len(le.classes_)
    logger.info(
        f"  {n_classes} classes · random baseline = {1/n_classes:.3f}"
    )

    # ── Embeddings (single pass per batch, with disk cache) ─────────────────
    cache = _cache_path(args.out)
    loaded = None if args.force_extraction else _load_cache(cache, layers_sorted, args.region)

    if loaded is not None:
        buffers, y, classes = loaded
        n_classes = len(classes)
        logger.info(
            f"Loaded embeddings from cache ({cache}). "
            "Use --force_extraction to re-run forward passes."
        )
    else:
        logger.info(
            f"Extracting layers {layers_sorted} — single forward pass per batch "
            f"(batch_size={args.batch_size}) ..."
        )
        buffers = extract_all_layers(
            model, tokenizer, subset, layers_sorted,
            args.region, device, bos_offset, args.batch_size,
        )
        _save_cache(cache, buffers, np.asarray(y), list(le.classes_), layers_sorted, args.region)

    # ── Classification ───────────────────────────────────────────────────────
    logger.info(f"Running {args.n_folds}-fold stratified CV ...")
    results = []
    t_cv_start = time.perf_counter()
    for layer_idx in layers_sorted:
        t_layer = time.perf_counter()
        X = buffers[layer_idx]
        valid = ~np.isnan(X[:, 0])
        n_dropped = int((~valid).sum())
        if n_dropped:
            logger.warning(f"  Layer {layer_idx}: dropping {n_dropped} rows with NaN embeddings")

        metrics = evaluate_layer(X[valid], y[valid], args.n_folds)
        metrics["layer"] = layer_idx
        results.append(metrics)
        logger.info(
            f"  Layer {layer_idx:>2}  "
            f"F1 = {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}  "
            f"Acc = {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}  "
            f"({time.perf_counter() - t_layer:.1f}s)"
        )
    logger.info(f"CV complete: {time.perf_counter() - t_cv_start:.1f}s total")

    # ── Outputs ──────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    csv_path = os.path.join(args.out, "layer_eval_results.csv")
    pd.DataFrame(results)[
        ["layer", "f1_mean", "f1_std", "accuracy_mean", "accuracy_std"]
    ].to_csv(csv_path, index=False)
    logger.info(f"Results table saved to {csv_path}")

    plot_path = os.path.join(args.out, "layer_eval_f1.png")
    plot_results(results, plot_path, args.n_folds, n_classes, num_blocks, args.region)

    best = max(results, key=lambda r: r["f1_mean"])
    logger.info(
        f"\n{'='*55}\n"
        f"Best layer : {best['layer']}  "
        f"(F1={best['f1_mean']:.3f} ± {best['f1_std']:.3f})\n"
        f"Use it with:\n"
        f"  python embeddings/esm/generate_embeddings.py --layer {best['layer']}\n"
        f"{'='*55}"
    )


if __name__ == "__main__":
    main()
