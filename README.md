# tcr_peptide_interactions

Scripts for pulling and normalizing public TCR–peptide interaction data.

## Data

Bulk data files are **not** committed (the IEDB MHC-ligand export alone is 8+ GB).
Regenerate them locally:

```bash
python fetch_iedb.py            # IEDB bulk exports -> ./iedb_data/
python fetch_vdjdb.py           # latest VDJdb release -> ./vdjdb_data/
python tsv_to_csv.py            # vdjdb_data/*.txt -> vdjdb_csv/*.csv
python build_dataset.py         # cleaned VDJdb, IEDB, McPAS, and combined TRB datasets -> ./processed/
```

### McPAS-TCR source

- Official site used for download: `https://friedmanlab.weizmann.ac.il/McPAS-TCR/`
- Local file expected by the pipeline: `./McPAS-TCR.csv`

## Suggested workflow

1. Fetch raw data with the scripts above.
2. Explore schemas and prototype cleaning rules in `notebooks/data_cleaning_visualization.ipynb`.
3. Move stable cleaning logic into `build_dataset.py`.
4. Write reproducible outputs to `processed/`.

Detailed writeups:

- `docs/cleaning_overview.md`
- `docs/vdjdb_cleaning_summary.md`
- `docs/iedb_cleaning_summary.md`
- `docs/combined_dataset_summary.md`

## Current cleaning rules

`build_dataset.py` currently builds:

- `processed/vdjdb_trb_clean.csv`
- `processed/iedb_trb_clean.csv`
- `processed/mcpas_trb_clean.csv`
- `processed/combined_trb_clean.csv`

All three outputs use a standardized schema:

- `cdr3`
- `tcr_chain`
- `v_gene`
- `j_gene`
- `peptide`
- `mhc_a`
- `mhc_b`
- `mhc_class`
- `source`

The VDJdb path applies these filters to `vdjdb_csv/vdjdb.csv`:

- keep only `HomoSapiens`
- keep only the requested TCR chain (`TRB` by default)
- require non-empty TCR, peptide, HLA, and V/J fields
- drop TCR or peptide sequences containing wildcard characters `X`, `*`, or `?`
- drop TCRs that map to more than one distinct peptide
- drop duplicate rows after filtering

The IEDB path applies these filters to `iedb_data/tcr_full_v3.csv`:

- keep only beta-chain TCR rows from `Chain 2`
- keep only human TCR rows based on the chain organism IRI
- use curated CDR3/V/J values when available, otherwise calculated values
- require non-empty `cdr3`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, and inferred `mhc_class`
- drop wildcard `cdr3` or peptide sequences containing `X`, `*`, or `?`
- drop TCRs that map to more than one distinct peptide
- drop duplicate rows after filtering

The McPAS path applies these filters to `McPAS-TCR.csv`:

- keep only `Human`
- use `CDR3.beta.aa`, `TRBV`, `TRBJ`, `Epitope.peptide`, and `MHC`
- infer `mhc_class` from the `MHC` field
- require non-empty `cdr3`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, and inferred `mhc_class`
- drop wildcard `cdr3` or peptide sequences containing `X`, `*`, or `?`
- drop TCRs that map to more than one distinct peptide
- drop duplicate rows after filtering

The combined output is a row-wise concatenation of the cleaned VDJdb, IEDB, and McPAS datasets, with `source` set to `VDJdb`, `IEDB`, or `McPAS`.

Sources:
- IEDB: https://www.iedb.org/database_export_v3.php
- VDJdb: https://github.com/antigenomics/vdjdb-db/releases
- McPAS-TCR: https://friedmanlab.weizmann.ac.il/McPAS-TCR/
