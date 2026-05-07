#!/usr/bin/env python3
"""Build cleaned TCR-peptide datasets from local VDJdb, IEDB, and McPAS exports.

Outputs (written to processed/):
    vdjdb_trb_clean.csv
    iedb_trb_clean.csv
    mcpas_trb_clean.csv
    combined_trb_clean.csv   row-wise concat of the three sources, with a 'source' column

All three sources apply a shared cleaning pipeline:
  - Human / HomoSapiens rows only
  - TRB chain only
  - Drop rows with missing required fields (cdr3, v_gene, j_gene, peptide, mhc_a)
  - Drop CDR3 or peptide sequences containing wildcard characters (X, *, ?)
  - Standardise V/J gene names to IMGT allele format via tidytcells;
    reject non-functional alleles (ORF, pseudogene) with enforce_functional=True —
    these gene reference sequences can contain in-frame stop codons that corrupt
    downstream sequence reconstruction via stitchr
  - Deduplicate

Source-specific additions:
  VDJdb  — keep only rows with vdjdb.score >= 1 (at least one independent
            verification beyond the original submission)
  IEDB   — keep only rows associated with a high-confidence assay method
            (multimer/tetramer, SPR, or x-ray crystallography) when
            iedb_data/tcell_full_v3.csv is present
"""

import argparse
import re
from pathlib import Path
from typing import cast

import pandas as pd
import tidytcells as tt

VDJDB_REQUIRED_COLUMNS = [
    "gene",
    "cdr3",
    "v.segm",
    "j.segm",
    "species",
    "mhc.a",
    "mhc.b",
    "mhc.class",
    "antigen.epitope",
]
STANDARD_COLUMNS = [
    "cdr3",
    "tcr_chain",
    "v_gene",
    "j_gene",
    "peptide",
    "mhc_a",
    "mhc_b",
    "mhc_class",
]
IEDB_REQUIRED_COLUMNS = [
    "cdr3",
    "tcr_chain",
    "v_gene",
    "j_gene",
    "peptide",
    "mhc_a",
    "mhc_class",
]


def load_vdjdb(vdjdb_csv: Path) -> pd.DataFrame:
    """Load the primary VDJdb table."""
    return pd.read_csv(vdjdb_csv, low_memory=False)


def load_iedb(iedb_csv: Path) -> pd.DataFrame:
    """Load the IEDB TCR export with its two-row header."""
    return pd.read_csv(iedb_csv, header=[0, 1], low_memory=False)


def load_iedb_methods(tcell_csv: Path) -> dict[tuple[str, str], frozenset[str]]:
    """Build an (epitope_iri, reference_iri) -> assay_methods lookup from the tcell export.

    The TCR export and tcell export share Epitope and Reference IRIs, making them
    joinable to recover the assay method (multimer/tetramer, ELISPOT, etc.) that
    is absent from the TCR-specific file.
    """
    df = pd.read_csv(tcell_csv, header=[0, 1], low_memory=False)

    def _norm(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.replace("https://", "http://", regex=False).str.strip()

    flat = pd.DataFrame({
        "epi": _norm(df[("Epitope", "IEDB IRI")]),
        "ref": _norm(df[("Reference", "IEDB IRI")]),
        "method": df[("Assay", "Method")],
    })
    return cast(
        dict[tuple[str, str], frozenset[str]],
        flat.groupby(["epi", "ref"])["method"]
        .apply(lambda x: frozenset(x.dropna()))
        .to_dict(),
    )


def load_mcpas(mcpas_csv: Path) -> pd.DataFrame:
    """Load the McPAS-TCR CSV export."""
    return pd.read_csv(mcpas_csv, na_values=["NA"], keep_default_na=True, low_memory=False)


def _normalize_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def _drop_missing_required(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col in required_columns:
        mask &= df[col].ne("")
    return df[mask].copy()


def _drop_wildcards(df: pd.DataFrame, cdr3_col: str, peptide_col: str) -> pd.DataFrame:
    wildcard = r"[X*?]"
    mask = ~df[cdr3_col].str.contains(wildcard, regex=True)
    mask &= ~df[peptide_col].str.contains(wildcard, regex=True)
    return df[mask].copy()


_COLON_ALLELE = re.compile(r":(\d)")
_AMBIGUOUS_SPLIT = re.compile(r"(?i)/.*|\s+or\s+.*")


def _preclean_gene_symbol(s: str) -> str:
    """Normalise a raw gene symbol string before passing it to tidytcells.

    Handles three fixable classes:
      - Colon-as-allele-separator  (TRBV1-4:01  → TRBV1-4*01)
      - Ambiguous slash calls       (TRBV12-3/4  → TRBV12-3)
      - "or"-separated ambiguity    (TCRBV3-1 or TCRBV3-2 → TCRBV3-1)
    Cross-chain contamination (TRBV29/DV5) becomes TRBV29 after slash
    stripping, which tidytcells then rejects as non-existent.
    Biologically impossible J-genes and true typos are left for tidytcells
    to reject via on_fail="reject".
    """
    if not s:
        return s
    s = _COLON_ALLELE.sub(r"*\1", s)
    s = _AMBIGUOUS_SPLIT.sub("", s).strip()
    return s


def _preclean_genes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("v_gene", "j_gene"):
        df[col] = df[col].map(_preclean_gene_symbol)
    return df


def _drop_promiscuous_tcrs(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop all rows for any CDR3 that maps to more than one distinct peptide.

    Cross-reactivity is real biology, so this filter is opt-in (--drop_promiscuous).
    Applied to the combined dataset after merging all sources, so a CDR3 that
    appears with two peptides across different databases is also caught.
    """
    pep_counts = df.groupby("cdr3")["peptide"].nunique()
    promiscuous = pep_counts[pep_counts > 1].index
    mask = df["cdr3"].isin(promiscuous)
    return df[~mask].copy(), int(mask.sum())


def _standardize_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize v_gene and j_gene to IMGT allele-level names via tidytcells.

    Rows where standardization fails (biologically impossible genes, true
    typos that survive preclean, cross-chain contamination, or ORF/pseudogene
    annotations) are dropped. enforce_functional=True rejects ORF and
    pseudogene alleles whose reference sequences can contain stop codons.
    """
    for col in ("v_gene", "j_gene"):
        df[col] = df[col].map(
            lambda s: tt.tr.standardize(
                s, precision="allele", on_fail="reject", log_failures=False,
                enforce_functional=True,
            ) if s else None
        )
    return df.dropna(subset=["v_gene", "j_gene"]).copy()



def _infer_mhc_class_from_alleles(alleles: pd.Series) -> pd.Series:
    s = alleles.fillna("").astype(str).str.strip()
    upper = s.str.upper()
    is_class_i = upper.str.contains(r"HLA CLASS I|HLA-[ABCEXFG]\b|B2M|HLA-E|HLA-C|HLA-A|HLA-B", regex=True)
    is_class_ii = upper.str.contains(r"HLA CLASS II|HLA-DP|HLA-DQ|HLA-DR", regex=True)
    out = pd.Series("", index=s.index, dtype="object")
    out.loc[is_class_i & ~is_class_ii] = "MHCI"
    out.loc[is_class_ii & ~is_class_i] = "MHCII"
    return out


def standardize_vdjdb(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "gene": "tcr_chain",
            "v.segm": "v_gene",
            "j.segm": "j_gene",
            "antigen.epitope": "peptide",
            "mhc.a": "mhc_a",
            "mhc.b": "mhc_b",
            "mhc.class": "mhc_class",
        }
    ).copy()
    return out[STANDARD_COLUMNS].reset_index(drop=True)


def _choose_nonempty(df: pd.DataFrame, first: tuple[str, str], second: tuple[str, str]) -> pd.Series:
    a = df[first].fillna("").astype(str).str.strip()
    b = df[second].fillna("").astype(str).str.strip()
    return a.where(a.ne(""), b)


def standardize_iedb(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "cdr3": _choose_nonempty(df, ("Chain 2", "CDR3 Curated"), ("Chain 2", "CDR3 Calculated")),
            "tcr_chain": "TRB",
            "v_gene": _choose_nonempty(df, ("Chain 2", "Curated V Gene"), ("Chain 2", "Calculated V Gene")),
            "j_gene": _choose_nonempty(df, ("Chain 2", "Curated J Gene"), ("Chain 2", "Calculated J Gene")),
            "peptide": df[("Epitope", "Name")].fillna("").astype(str).str.strip(),
            "mhc_a": df[("Assay", "MHC Allele Names")].fillna("").astype(str).str.strip(),
            "mhc_b": "",
        }
    )
    out["mhc_class"] = _infer_mhc_class_from_alleles(out["mhc_a"])
    return out[STANDARD_COLUMNS].reset_index(drop=True)


def standardize_mcpas(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "cdr3": df["CDR3.beta.aa"].fillna("").astype(str).str.strip(),
            "tcr_chain": "TRB",
            "v_gene": df["TRBV"].fillna("").astype(str).str.strip(),
            "j_gene": df["TRBJ"].fillna("").astype(str).str.strip(),
            "peptide": df["Epitope.peptide"].fillna("").astype(str).str.strip(),
            "mhc_a": df["MHC"].fillna("").astype(str).str.strip(),
            "mhc_b": "",
        }
    )
    out["mhc_class"] = _infer_mhc_class_from_alleles(out["mhc_a"])
    return out[STANDARD_COLUMNS].reset_index(drop=True)


def clean_vdjdb(df: pd.DataFrame, chain: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean a raw VDJdb table to the project standard schema.

    Filters applied in order:
      1. HomoSapiens only
      2. Requested chain (default TRB)
      3. Drop rows missing any required field
      4. Drop CDR3 / peptide sequences containing X, *, or ?
      5. vdjdb.score >= 1 (at least one independent verification method)
      6. Standardise V/J to IMGT allele names; drop non-functional alleles
      7. Deduplicate
    """
    stats: dict[str, int] = {"loaded_rows": len(df)}
    cleaned = df.copy()
    cleaned = _normalize_text(cleaned, VDJDB_REQUIRED_COLUMNS)

    cleaned = cleaned[cleaned["species"].eq("HomoSapiens")].copy()
    stats["after_species"] = len(cleaned)

    cleaned = cleaned[cleaned["gene"].eq(chain)].copy()
    stats["after_chain"] = len(cleaned)

    cleaned = _drop_missing_required(cleaned, VDJDB_REQUIRED_COLUMNS)
    stats["after_required_fields"] = len(cleaned)

    cleaned = _drop_wildcards(cleaned, "cdr3", "antigen.epitope")
    stats["after_wildcards"] = len(cleaned)

    # Keep only annotations backed by at least one independent verification method
    # (score 0 = single low-confidence assay; 1-3 = progressively more evidence).
    if "vdjdb.score" in cleaned.columns:
        cleaned = cleaned[cleaned["vdjdb.score"].ge(1)].copy()
    stats["after_confidence_filter"] = len(cleaned)

    standardized = standardize_vdjdb(cleaned)
    standardized = _preclean_genes(standardized)
    standardized = _standardize_genes(standardized)
    stats["after_gene_standardization"] = len(standardized)
    standardized = standardized.drop_duplicates(subset=STANDARD_COLUMNS).copy()
    stats["after_dedup"] = len(standardized)
    return standardized, stats


IEDB_HIGH_CONF_METHODS: frozenset[str] = frozenset({
    "multimer/tetramer",
    "surface plasmon resonance (SPR)",
    "x-ray crystallography",
})


def clean_iedb(
    df: pd.DataFrame,
    method_map: dict[tuple[str, str], frozenset[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the project cleaning rules to IEDB TCR export rows.

    method_map: optional output of load_iedb_methods(). When provided, only
    rows associated with a high-confidence assay (multimer/tetramer, SPR, or
    x-ray crystallography) are kept.
    """
    stats: dict[str, int] = {"loaded_rows": len(df)}
    cleaned = df.copy()

    cleaned = cleaned[
        cleaned[("Chain 2", "Organism IRI")].fillna("").astype(str).str.contains("NCBITaxon_9606", na=False)
    ].copy()
    stats["after_species"] = len(cleaned)

    cleaned = cleaned[
        cleaned[("Chain 2", "Type")].fillna("").astype(str).str.strip().eq("beta")
    ].copy()
    stats["after_chain"] = len(cleaned)

    if method_map is not None:
        def _norm(s: pd.Series) -> pd.Series:
            return s.fillna("").astype(str).str.replace("https://", "http://", regex=False).str.strip()

        epi = _norm(cleaned[("Epitope", "IEDB IRI")])
        ref = _norm(cleaned[("Reference", "IEDB IRI")])
        mask = [
            bool(method_map.get((e, r), frozenset()) & IEDB_HIGH_CONF_METHODS)
            for e, r in zip(epi, ref)
        ]
        cleaned = cleaned[mask].copy()
        stats["after_confidence_filter"] = len(cleaned)

    standardized = standardize_iedb(cleaned)
    standardized = _normalize_text(standardized, STANDARD_COLUMNS)

    standardized = _drop_missing_required(standardized, IEDB_REQUIRED_COLUMNS)
    stats["after_required_fields"] = len(standardized)

    standardized = _drop_wildcards(standardized, "cdr3", "peptide")
    stats["after_wildcards"] = len(standardized)

    standardized = _preclean_genes(standardized)
    standardized = _standardize_genes(standardized)
    stats["after_gene_standardization"] = len(standardized)
    standardized = standardized.drop_duplicates(subset=STANDARD_COLUMNS).copy()
    stats["after_dedup"] = len(standardized)

    return standardized.reset_index(drop=True), stats


def clean_mcpas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean a raw McPAS-TCR table to the project standard schema.

    Filters applied in order:
      1. Human only
      2. Drop rows missing any required field
      3. Drop CDR3 / peptide sequences containing X, *, or ?
      4. Standardise V/J to IMGT allele names; drop non-functional alleles
      5. Deduplicate

    No assay-confidence filter is applied: McPAS's identification method
    field uses an opaque numeric codebook with no confirmed legend.
    """
    stats: dict[str, int] = {"loaded_rows": len(df)}
    cleaned = df.copy()

    cleaned = cleaned[cleaned["Species"].fillna("").astype(str).str.strip().eq("Human")].copy()
    stats["after_species"] = len(cleaned)

    standardized = standardize_mcpas(cleaned)
    standardized = _normalize_text(standardized, STANDARD_COLUMNS)

    standardized = _drop_missing_required(standardized, IEDB_REQUIRED_COLUMNS)
    stats["after_required_fields"] = len(standardized)

    standardized = _drop_wildcards(standardized, "cdr3", "peptide")
    stats["after_wildcards"] = len(standardized)

    standardized = _preclean_genes(standardized)
    standardized = _standardize_genes(standardized)
    stats["after_gene_standardization"] = len(standardized)
    standardized = standardized.drop_duplicates(subset=STANDARD_COLUMNS).copy()
    stats["after_dedup"] = len(standardized)

    return standardized.reset_index(drop=True), stats


def print_stats(label: str, stats: dict[str, int]) -> None:
    print(f"\n[{label}]")
    print(f"Loaded rows:                     {stats['loaded_rows']:,}")
    if "after_species" in stats:
        print(f"After Homo sapiens filter:      {stats['after_species']:,}")
    if "after_chain" in stats:
        print(f"After chain filter:             {stats['after_chain']:,}")
    if "after_confidence_filter" in stats:
        print(f"After confidence filter:        {stats['after_confidence_filter']:,}")
    print(f"After required-field filter:    {stats['after_required_fields']:,}")
    print(f"After wildcard filter:          {stats['after_wildcards']:,}")
    if "after_gene_standardization" in stats:
        print(f"After gene std + functional:    {stats['after_gene_standardization']:,}")
    print(f"After de-duplication:           {stats['after_dedup']:,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vdjdb-csv",
        type=Path,
        default=Path("vdjdb_csv/vdjdb.csv"),
        help="path to the main VDJdb CSV",
    )
    ap.add_argument(
        "--iedb-csv",
        type=Path,
        default=Path("iedb_data/tcr_full_v3.csv"),
        help="path to the IEDB TCR CSV export",
    )
    ap.add_argument(
        "--iedb-tcell-csv",
        type=Path,
        default=Path("iedb_data/tcell_full_v3.csv"),
        help="path to the IEDB full T cell assay export (used for assay-method confidence filter)",
    )
    ap.add_argument(
        "--chain",
        default="TRB",
        choices=["TRA", "TRB"],
        help="TCR chain to retain (default: TRB)",
    )
    ap.add_argument(
        "--mcpas-csv",
        type=Path,
        default=Path("mcpas_data/McPAS-TCR.csv"),
        help="path to the McPAS-TCR CSV export",
    )
    ap.add_argument(
        "--vdjdb-out",
        type=Path,
        default=Path("processed/vdjdb_trb_clean.csv"),
        help="output path for the cleaned VDJdb dataset",
    )
    ap.add_argument(
        "--iedb-out",
        type=Path,
        default=Path("processed/iedb_trb_clean.csv"),
        help="output path for the cleaned IEDB dataset",
    )
    ap.add_argument(
        "--combined-out",
        type=Path,
        default=Path("processed/combined_trb_clean.csv"),
        help="output path for the combined dataset",
    )
    ap.add_argument(
        "--mcpas-out",
        type=Path,
        default=Path("processed/mcpas_trb_clean.csv"),
        help="output path for the cleaned McPAS dataset",
    )
    ap.add_argument(
        "--drop_promiscuous",
        action="store_true",
        default=False,
        help=(
            "drop all rows for any CDR3 that maps to more than one distinct peptide "
            "in the combined dataset (default: keep — cross-reactivity is real biology)"
        ),
    )
    args = ap.parse_args()

    vdjdb_df = load_vdjdb(args.vdjdb_csv)
    vdjdb_clean, vdjdb_stats = clean_vdjdb(vdjdb_df, chain=args.chain)
    vdjdb_clean = vdjdb_clean.assign(source="VDJdb")

    iedb_df = load_iedb(args.iedb_csv)
    iedb_method_map = load_iedb_methods(args.iedb_tcell_csv) if args.iedb_tcell_csv.exists() else None
    if iedb_method_map is None:
        print(f"Warning: {args.iedb_tcell_csv} not found — skipping IEDB confidence filter.")
    iedb_clean, iedb_stats = clean_iedb(iedb_df, method_map=iedb_method_map)
    iedb_clean = iedb_clean.assign(source="IEDB")

    mcpas_df = load_mcpas(args.mcpas_csv)
    mcpas_clean, mcpas_stats = clean_mcpas(mcpas_df)
    mcpas_clean = mcpas_clean.assign(source="McPAS")

    combined = pd.concat([vdjdb_clean, iedb_clean, mcpas_clean], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=STANDARD_COLUMNS + ["source"]
    ).reset_index(drop=True)

    if args.drop_promiscuous:
        combined, n_dropped = _drop_promiscuous_tcrs(combined)
        print(f"Promiscuous TCRs dropped:        {n_dropped:,} rows")

    for path in [args.vdjdb_out, args.iedb_out, args.mcpas_out, args.combined_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    vdjdb_clean.to_csv(args.vdjdb_out, index=False)
    iedb_clean.to_csv(args.iedb_out, index=False)
    mcpas_clean.to_csv(args.mcpas_out, index=False)
    combined.to_csv(args.combined_out, index=False)

    print_stats(f"VDJdb ({args.vdjdb_csv})", vdjdb_stats)
    print_stats(f"IEDB ({args.iedb_csv})", iedb_stats)
    print_stats(f"McPAS ({args.mcpas_csv})", mcpas_stats)
    print(f"\n[combined]")
    print(f"Rows written:                    {len(combined):,}")
    print(f"VDJdb rows:                      {len(vdjdb_clean):,}")
    print(f"IEDB rows:                       {len(iedb_clean):,}")
    print(f"McPAS rows:                      {len(mcpas_clean):,}")
    print(f"Wrote cleaned VDJdb dataset to {args.vdjdb_out}")
    print(f"Wrote cleaned IEDB dataset to {args.iedb_out}")
    print(f"Wrote cleaned McPAS dataset to {args.mcpas_out}")
    print(f"Wrote combined dataset to {args.combined_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
