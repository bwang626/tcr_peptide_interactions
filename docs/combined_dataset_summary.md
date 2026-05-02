# Combined Dataset Summary

## Inputs to pooling

- `processed/vdjdb_trb_clean.csv`: `83,644` rows
- `processed/iedb_trb_clean.csv`: `144,443` rows

## Pooling behavior

- The cleaned VDJdb and IEDB datasets are concatenated row-wise.
- Both datasets already share the same standardized schema.
- A `source` column is retained so each row can still be traced back to `VDJdb` or `IEDB`.

## What was removed during pooling

- Exact cross-source duplicate rows on the standardized biological columns: `0`
- Sum of individual cleaned rows: `228,087`
- Final pooled rows: `228,087`

So in the current build, pooling did not remove any extra rows beyond the dataset-specific cleaning steps.

## Final combined dataset

- File: `processed/combined_trb_clean.csv`
- Rows: `228,087`
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

## Source composition

| Source | Rows |
| --- | ---: |
| IEDB | 144,443 |
| VDJdb | 83,644 |

## What the combined dataset represents

Each row in the final combined dataset is a cleaned TRB TCR-peptide interaction record with:

- beta-chain CDR3 sequence
- V gene
- J gene
- peptide / epitope sequence
- MHC allele information
- MHC class
- source dataset label
