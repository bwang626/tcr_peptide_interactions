# Cleaning Overview

This project currently builds three cleaned datasets:

- `processed/vdjdb_trb_clean.csv`
- `processed/iedb_trb_clean.csv`
- `processed/combined_trb_clean.csv`

The overall logic is:

1. Pull raw VDJdb and IEDB data locally.
2. Extract a shared TRB-focused schema.
3. Remove rows that violate the cleaning rules.
4. Save cleaned per-source outputs.
5. Pool the cleaned outputs into one combined dataset with a `source` column.

## Shared modeling-facing columns

- `cdr3`
- `tcr_chain`
- `v_gene`
- `j_gene`
- `peptide`
- `mhc_a`
- `mhc_b`
- `mhc_class`
- `source`

## Shared cleaning ideas

Across both datasets, the cleaner attempts to enforce:

- human-only rows
- beta-chain TRB records
- non-empty required fields
- no wildcard sequences in TCR or peptide
- one TCR identity mapping to one peptide
- no duplicate rows

## Important source-specific difference

IEDB does not expose a second MHC field comparable to VDJdb's `mhc_b`, so `mhc_b` is blank in the cleaned IEDB output.
