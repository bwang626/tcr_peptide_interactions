# VDJdb Cleaning Summary

## Raw source

- File: `vdjdb_csv/vdjdb.csv`
- Raw rows loaded: `226,494`

## Raw columns pulled into the cleaned dataset

- `cdr3` -> CDR3 amino-acid sequence
- `gene` -> TCR chain label
- `v.segm` -> V gene
- `j.segm` -> J gene
- `antigen.epitope` -> peptide sequence
- `mhc.a` -> primary HLA / MHC field
- `mhc.b` -> secondary MHC-associated field
- `mhc.class` -> MHC class
- `species` -> used only as a filter

## Cleaning steps

1. Keep only `HomoSapiens` rows.
2. Keep only `TRB` rows.
3. Require non-empty `cdr3`, `v.segm`, `j.segm`, `mhc.a`, `mhc.b`, `mhc.class`, and `antigen.epitope`.
4. Remove wildcard sequences where `cdr3` or `antigen.epitope` contains `X`, `*`, or `?`.
5. Remove TCRs that bind more than one distinct peptide.
6. Drop duplicate rows.

For the multi-peptide rule, TCR identity is defined as:
`(gene, cdr3, v.segm, j.segm)`.

## What remained after each step

| Stage | Rows remaining | Rows removed at step |
| --- | ---: | ---: |
| Loaded raw VDJdb rows | 226,494 | 0 |
| After HomoSapiens filter | 206,106 | 20,388 |
| After TRB filter | 113,281 | 92,825 |
| After required-field filter | 113,281 | 0 |
| After wildcard filter | 113,281 | 0 |
| After multi-peptide filter | 101,798 | 11,483 |
| After de-duplication | 83,644 | 18,154 |

## Final cleaned VDJdb dataset

- File: `processed/vdjdb_trb_clean.csv`
- Rows: `83,644`
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
- `source` is always `VDJdb`
