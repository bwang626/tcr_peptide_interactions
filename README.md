# tcr_peptide_interactions

Scripts for pulling and normalizing public TCR–peptide interaction data.

## Data

Bulk data files are **not** committed (the IEDB MHC-ligand export alone is 8+ GB).
Regenerate them locally:

```bash
python fetch_iedb.py            # IEDB bulk exports -> ./iedb_data/
python fetch_vdjdb.py           # latest VDJdb release -> ./vdjdb_data/
python tsv_to_csv.py            # vdjdb_data/*.txt -> vdjdb_csv/*.csv
```

Sources:
- IEDB: https://www.iedb.org/database_export_v3.php
- VDJdb: https://github.com/antigenomics/vdjdb-db/releases
