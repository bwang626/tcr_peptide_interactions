"""
explore_embeddings.py

Cluster a random subset of TCR embeddings and visualise them in 2D via UMAP.

Two panels are produced side-by-side:
  Left  — each point coloured by its ground-truth peptide epitope label.
  Right — each point coloured by its unsupervised cluster assignment.

Comparing the two panels directly shows how well the ESM embedding separates
binding groups without any supervision.  The Adjusted Rand Index (ARI) printed
at the end quantifies cluster-vs-label agreement (1 = perfect, 0 = random).

Dependencies (beyond the project base):
    pip install umap-learn hdbscan

Run from the repo root:
    python embeddings/esm/explore_embeddings.py
    python embeddings/esm/explore_embeddings.py --n 2000 --embedding full
    python embeddings/esm/explore_embeddings.py --n 5000 --cluster kmeans --k 8
    python embeddings/esm/explore_embeddings.py --n 3000 --stratify

Outputs (outputs/embeddings/esm/):
    umap_<embedding>_n<N>.png        scatter plot (peptide labels | cluster labels)
    umap_<embedding>_n<N>_coords.csv 2-D coordinates + labels for further analysis
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_OUT    = "outputs/embeddings/esm"
DEFAULT_EMB = "cdr3"
DEFAULT_N   = 5000


# ---------------------------------------------------------------------------
# Lazy imports with friendly error messages
# ---------------------------------------------------------------------------

def _import_umap():
    try:
        import umap
        return umap.UMAP
    except ImportError:
        sys.exit("umap-learn is required: pip install umap-learn")


def _import_hdbscan():
    try:
        import hdbscan
        return hdbscan.HDBSCAN
    except ImportError:
        sys.exit("hdbscan is required: pip install hdbscan")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(
    emb_dir: str,
    embedding: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the requested .npy file and the matching embedding_index.csv."""
    emb_file   = os.path.join(emb_dir, f"{embedding}_embeddings.npy")
    index_file = os.path.join(emb_dir, "embedding_index.csv")

    for path in (emb_file, index_file):
        if not os.path.exists(path):
            sys.exit(
                f"File not found: {path}\n"
                "Run generate_embeddings.py first to produce the embedding files."
            )

    embs  = np.load(emb_file)
    index = pd.read_csv(index_file, index_col=0)

    if len(embs) != len(index):
        sys.exit(
            f"Shape mismatch: {emb_file} has {len(embs)} rows but "
            f"embedding_index.csv has {len(index)} rows."
        )

    return embs, index


# ---------------------------------------------------------------------------
# Subset sampling
# ---------------------------------------------------------------------------

def sample_subset(
    embs: np.ndarray,
    index: pd.DataFrame,
    n: int,
    stratify: bool,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Sample n rows.  If stratify=True, draw equally from each peptide class."""
    rng = np.random.default_rng(seed)

    # Drop rows with NaN embeddings or missing peptide
    valid = ~np.isnan(embs[:, 0]) & index["peptide"].notna()
    embs  = embs[valid]
    index = index[valid].reset_index(drop=True)

    if n >= len(index):
        logger.info(f"  Requested n={n} ≥ available {len(index)} — using all rows")
        return embs, index

    if stratify:
        peptides = index["peptide"].unique()
        per_class = max(1, n // len(peptides))
        parts_e, parts_i = [], []
        for pep in peptides:
            mask = (index["peptide"] == pep).values
            idx  = np.where(mask)[0]
            chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
            parts_e.append(embs[chosen])
            parts_i.append(index.iloc[chosen])
        sel_embs  = np.vstack(parts_e)
        sel_index = pd.concat(parts_i, ignore_index=True)
        logger.info(
            f"  Stratified sample: {len(sel_index)} rows "
            f"(~{per_class} per class × {len(peptides)} peptides)"
        )
    else:
        chosen    = rng.choice(len(index), size=n, replace=False)
        sel_embs  = embs[chosen]
        sel_index = index.iloc[chosen].reset_index(drop=True)
        logger.info(f"  Random sample: {len(sel_index)} rows")

    return sel_embs, sel_index


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    """Reduce X (N, D) → (N, 2) with UMAP."""
    UMAP = _import_umap()
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=False,
    )
    logger.info(
        f"Running UMAP  (n={len(X)}, D={X.shape[1]}, "
        f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}) ..."
    )
    coords = reducer.fit_transform(X)
    logger.info("  UMAP complete")
    return coords.astype(np.float32)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_hdbscan(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    HDBSCAN = _import_hdbscan()
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(5, min_cluster_size // 5),
        metric="euclidean",     # applied to UMAP coords, not raw embeddings
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
    return labels


def run_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    logger.info(f"  KMeans: {k} clusters")
    return labels


def cluster(
    coords: np.ndarray,
    method: str,
    k: int | None,
    min_cluster_size: int,
    seed: int,
) -> np.ndarray:
    if method == "hdbscan":
        return run_hdbscan(coords, min_cluster_size)
    if method == "kmeans":
        if k is None:
            sys.exit("--k is required when using --cluster kmeans")
        return run_kmeans(coords, k, seed)
    # method == "none"
    return np.zeros(len(coords), dtype=int)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_NOISE_COLOR = "#cccccc"


def _scatter_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    labels,
    title: str,
    categorical: bool,
    alpha: float = 0.55,
    point_size: float = 4.0,
) -> None:
    """Draw one UMAP scatter panel."""
    if categorical:
        unique = [l for l in sorted(set(labels)) if l != -1]
        cmap   = plt.cm.get_cmap("tab20", max(len(unique), 1))
        color_map = {l: cmap(i / max(len(unique) - 1, 1)) for i, l in enumerate(unique)}

        # Noise points first (grey, behind everything)
        noise = labels == -1
        if noise.any():
            ax.scatter(
                coords[noise, 0], coords[noise, 1],
                c=_NOISE_COLOR, s=point_size * 0.6, alpha=0.3,
                linewidths=0, rasterized=True, label="noise / unlabelled",
            )

        # One scatter call per label for legend
        for lbl in unique:
            mask = labels == lbl
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[color_map[lbl]], s=point_size, alpha=alpha,
                linewidths=0, rasterized=True,
                label=str(lbl) if len(str(lbl)) <= 20 else str(lbl)[:18] + "…",
            )

        # Annotate cluster centroids with their label
        for lbl in unique:
            mask = labels == lbl
            cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
            ax.text(
                cx, cy, str(lbl),
                fontsize=6, ha="center", va="center", fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="black")],
            )

        handles, lbls = ax.get_legend_handles_labels()
        if len(unique) <= 20:
            ax.legend(
                handles, lbls,
                markerscale=2, fontsize=6,
                loc="upper right", framealpha=0.7,
                handletextpad=0.3, borderpad=0.5,
            )
    else:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap="viridis", s=point_size,
            alpha=alpha, linewidths=0, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", "datalim")


def plot_umap(
    coords: np.ndarray,
    peptide_labels: np.ndarray,
    cluster_labels: np.ndarray,
    ari: float,
    n: int,
    embedding: str,
    cluster_method: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_panel(
        axes[0], coords, peptide_labels,
        title=f"Peptide epitope  (n={n:,})",
        categorical=True,
    )
    _scatter_panel(
        axes[1], coords, cluster_labels,
        title=f"Cluster ({cluster_method})   ARI vs peptide = {ari:.3f}",
        categorical=True,
    )

    fig.suptitle(
        f"ESMplusplus TCR Embeddings — {embedding} pooling\n"
        f"UMAP 2D projection  ·  n = {n:,}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb_dir",    default=BASE_OUT,
                   help="Directory containing .npy files and embedding_index.csv")
    p.add_argument("--embedding",  choices=["cdr3", "full"], default=DEFAULT_EMB,
                   help="Embedding type to load (default: cdr3)")
    p.add_argument("--n",          type=int, default=DEFAULT_N,
                   help="Number of TCRs to sample (default: 5000)")
    p.add_argument("--stratify",   action="store_true",
                   help="Sample equally from each peptide class instead of at random")
    p.add_argument("--cluster",    choices=["hdbscan", "kmeans", "none"],
                   default="hdbscan",
                   help="Clustering algorithm (default: hdbscan)")
    p.add_argument("--k",          type=int, default=None,
                   help="Number of clusters for kmeans (required with --cluster kmeans)")
    p.add_argument("--min_cluster_size", type=int, default=None,
                   help="HDBSCAN min_cluster_size (default: ~1%% of n)")
    p.add_argument("--n_neighbors", type=int, default=15,
                   help="UMAP n_neighbors (default: 15)")
    p.add_argument("--min_dist",   type=float, default=0.1,
                   help="UMAP min_dist (default: 0.1)")
    p.add_argument("--metric",     default="cosine",
                   help="UMAP distance metric (default: cosine)")
    p.add_argument("--out",        default=BASE_OUT,
                   help="Output directory (default: outputs/embeddings/esm)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.embedding} embeddings from {args.emb_dir} ...")
    embs, index = load_embeddings(args.emb_dir, args.embedding)
    logger.info(f"  {len(embs):,} rows, D={embs.shape[1]}")

    # ── Sample ───────────────────────────────────────────────────────────────
    embs, index = sample_subset(embs, index, args.n, args.stratify, args.seed)
    n = len(embs)

    peptide_labels = index["peptide"].fillna("unknown").values

    # ── UMAP ─────────────────────────────────────────────────────────────────
    coords = run_umap(embs, args.n_neighbors, args.min_dist, args.metric, args.seed)

    # ── Cluster ──────────────────────────────────────────────────────────────
    min_cs = args.min_cluster_size or max(20, n // 100)
    logger.info(f"Clustering ({args.cluster}) on UMAP coordinates ...")
    cluster_labels = cluster(coords, args.cluster, args.k, min_cs, args.seed)

    # ── ARI ──────────────────────────────────────────────────────────────────
    from sklearn.metrics import adjusted_rand_score
    # Exclude noise points (-1) from ARI calculation
    mask = cluster_labels != -1
    if mask.sum() > 0 and args.cluster != "none":
        ari = adjusted_rand_score(peptide_labels[mask], cluster_labels[mask])
    else:
        ari = float("nan")

    n_clusters  = len(set(cluster_labels) - {-1})
    n_noise     = int((cluster_labels == -1).sum())
    n_peptides  = len(set(peptide_labels))
    logger.info(
        f"\nResults\n"
        f"  Unique peptides   : {n_peptides}\n"
        f"  Clusters found    : {n_clusters}  (noise: {n_noise})\n"
        f"  ARI (vs peptide)  : {ari:.4f}  "
        f"[1=perfect match, 0=random, negative=worse than random]"
    )

    # ── Save coords ──────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.embedding}_n{n}"

    coords_df = index[["peptide"]].copy()
    coords_df["umap_1"]  = coords[:, 0]
    coords_df["umap_2"]  = coords[:, 1]
    coords_df["cluster"] = cluster_labels
    csv_path = os.path.join(args.out, f"umap_{tag}_coords.csv")
    coords_df.to_csv(csv_path, index=False)
    logger.info(f"Coordinates saved to {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out, f"umap_{tag}.png")
    plot_umap(
        coords, peptide_labels, cluster_labels,
        ari=ari, n=n,
        embedding=args.embedding,
        cluster_method=args.cluster,
        out_path=plot_path,
    )


if __name__ == "__main__":
    main()
