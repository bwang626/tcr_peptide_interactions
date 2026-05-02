# IEDB Cleaning Summary

## Raw source

- File: `iedb_data/tcr_full_v3.csv`
- Raw rows loaded: `226,280`

## Raw columns pulled into the cleaned dataset

- `('Chain 2', 'CDR3 Curated')` or `('Chain 2', 'CDR3 Calculated')` -> `cdr3`
- `('Chain 2', 'Curated V Gene')` or `('Chain 2', 'Calculated V Gene')` -> `v_gene`
- `('Chain 2', 'Curated J Gene')` or `('Chain 2', 'Calculated J Gene')` -> `j_gene`
- `('Epitope', 'Name')` -> `peptide`
- `('Assay', 'MHC Allele Names')` -> `mhc_a`
- inferred from `('Assay', 'MHC Allele Names')` -> `mhc_class`
- `('Chain 2', 'Type')` -> used to keep beta-chain TCR rows
- `('Chain 2', 'Organism IRI')` -> used to keep human rows

Notes:

- `tcr_chain` is standardized to `TRB`
- `mhc_b` is not available as a directly comparable field in IEDB, so it is left blank

## Cleaning steps

1. Keep only beta-chain rows from `Chain 2`.
2. Keep only human rows using `('Chain 2', 'Organism IRI')`.
3. Use curated CDR3/V/J values when present, otherwise calculated values.
4. Require non-empty `cdr3`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, and `mhc_class`.
5. Remove wildcard sequences where `cdr3` or `peptide` contains `X`, `*`, or `?`.
6. Remove TCRs that bind more than one distinct peptide.
7. Drop duplicate rows.

For the multi-peptide rule, TCR identity is defined as:
`(tcr_chain, cdr3, v_gene, j_gene)`.

## What remained after each step

| Stage | Rows remaining | Rows removed at step |
| --- | ---: | ---: |
| Loaded raw IEDB rows | 226,280 | 0 |
| After beta-chain filter | 195,028 | 31,252 |
| After human filter | 187,488 | 7,540 |
| After required-field filter | 160,630 | 26,858 |
| After wildcard filter | 160,626 | 4 |
| After multi-peptide filter | 146,819 | 13,807 |
| After de-duplication | 144,443 | 2,376 |

## Final cleaned IEDB dataset

- File: `processed/iedb_trb_clean.csv`
- Rows: `144,443`
- Columns:
  - `cdr3`
  - `tcr_chain`
  - `v_gene`
  - `j_gene`
  - `peptide`
  - `mhc_a`
  - `mhc_b`
  - `mhc_class`
  - `source`
- `source` is always `IEDB`
