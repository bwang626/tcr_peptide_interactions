#!/usr/bin/env python3
"""Build cleaned TCR-peptide datasets from local VDJdb, IEDB, and McPAS exports.

Use the notebook for exploratory cleaning and validation. Once the logic is
stable, keep the final reproducible pipeline here.
"""

import argparse
from pathlib import Path

import pandas as pd

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


def _drop_multi_peptide_tcrs(df: pd.DataFrame, tcr_key: list[str], peptide_col: str) -> pd.DataFrame:
    peptide_counts = df.groupby(tcr_key)[peptide_col].nunique()
    valid_keys = peptide_counts[peptide_counts == 1].index
    keep_mask = df.set_index(tcr_key).index.isin(valid_keys)
    return df[keep_mask].copy()


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
    """Apply the project cleaning rules."""
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

    cleaned = _drop_multi_peptide_tcrs(cleaned, ["gene", "cdr3", "v.segm", "j.segm"], "antigen.epitope")
    stats["after_multi_peptide_filter"] = len(cleaned)

    cleaned = cleaned.drop_duplicates(
        subset=["cdr3", "gene", "v.segm", "j.segm", "antigen.epitope", "mhc.a", "mhc.b", "mhc.class"]
    ).copy()
    stats["after_dedup"] = len(cleaned)

    return standardize_vdjdb(cleaned), stats


def clean_iedb(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the project cleaning rules to IEDB TCR export rows."""
    stats: dict[str, int] = {"loaded_rows": len(df)}
    cleaned = df.copy()

    cleaned = cleaned[
        cleaned[("Chain 2", "Type")].fillna("").astype(str).str.strip().eq("beta")
    ].copy()
    stats["after_chain"] = len(cleaned)

    cleaned = cleaned[
        cleaned[("Chain 2", "Organism IRI")].fillna("").astype(str).str.contains("NCBITaxon_9606", na=False)
    ].copy()
    stats["after_species"] = len(cleaned)

    standardized = standardize_iedb(cleaned)
    standardized = _normalize_text(standardized, STANDARD_COLUMNS)

    standardized = _drop_missing_required(standardized, IEDB_REQUIRED_COLUMNS)
    stats["after_required_fields"] = len(standardized)

    standardized = _drop_wildcards(standardized, "cdr3", "peptide")
    stats["after_wildcards"] = len(standardized)

    standardized = _drop_multi_peptide_tcrs(standardized, ["tcr_chain", "cdr3", "v_gene", "j_gene"], "peptide")
    stats["after_multi_peptide_filter"] = len(standardized)

    standardized = standardized.drop_duplicates(subset=STANDARD_COLUMNS).copy()
    stats["after_dedup"] = len(standardized)

    return standardized.reset_index(drop=True), stats


def clean_mcpas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the project cleaning rules to McPAS-TCR rows."""
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

    standardized = _drop_multi_peptide_tcrs(standardized, ["tcr_chain", "cdr3", "v_gene", "j_gene"], "peptide")
    stats["after_multi_peptide_filter"] = len(standardized)

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
    print(f"After required-field filter:    {stats['after_required_fields']:,}")
    print(f"After wildcard filter:          {stats['after_wildcards']:,}")
    print(f"After multi-peptide TCR filter: {stats['after_multi_peptide_filter']:,}")
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
        "--chain",
        default="TRB",
        choices=["TRA", "TRB"],
        help="TCR chain to retain (default: TRB)",
    )
    ap.add_argument(
        "--mcpas-csv",
        type=Path,
        default=Path("McPAS-TCR.csv"),
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
    args = ap.parse_args()

    vdjdb_df = load_vdjdb(args.vdjdb_csv)
    vdjdb_clean, vdjdb_stats = clean_vdjdb(vdjdb_df, chain=args.chain)
    vdjdb_clean = vdjdb_clean.assign(source="VDJdb")

    iedb_df = load_iedb(args.iedb_csv)
    iedb_clean, iedb_stats = clean_iedb(iedb_df)
    iedb_clean = iedb_clean.assign(source="IEDB")

    mcpas_df = load_mcpas(args.mcpas_csv)
    mcpas_clean, mcpas_stats = clean_mcpas(mcpas_df)
    mcpas_clean = mcpas_clean.assign(source="McPAS")

    combined = pd.concat([vdjdb_clean, iedb_clean, mcpas_clean], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=STANDARD_COLUMNS + ["source"]
    ).reset_index(drop=True)

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
