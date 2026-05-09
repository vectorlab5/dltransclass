# Dataset preparation

We do not redistribute the underlying corpora; please obtain them from the original sources cited in Section 4.1 of the paper. The pre-processing pipeline expects each dataset to be materialized as a parquet table under `data/processed/<dataset_name>/{train,val,test}.parquet` with the following schema:

| Column            | Type   | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| `title`           | str    | Title field. Empty string when missing.                  |
| `abstract`        | str    | Abstract field. Empty string when missing.               |
| `keywords`        | str    | Author or controlled-vocabulary keywords (semicolon-sep).|
| `subject`         | str    | Subject heading or category code.                        |
| `label`           | int    | Integer-encoded gold label (0..C-1).                     |
| `observed_title`  | int    | (optional) explicit 0/1 observation flag.                |
| ... per field     |        |                                                          |

Author-stratified splits are produced with `scripts/prepare_dataset.py` (to be added). Duplicate-title and near-duplicate abstract checks are applied before the final partitioning, as described in Section 4.1.

## Sources

| Dataset            | Source                                                                    |
|--------------------|---------------------------------------------------------------------------|
| UDC-OldBooks       | Kragelj & Kljajić Borštnar, *J. of Documentation*, 2021.                  |
| LCSH-Theses        | Usta, *Information Processing & Management*, 2025.                        |
| JUCS-Meta          | Mitra et al., *Knowledge-Based Systems*, 2025.                            |
| CompScholar-Meta   | Curated for this paper (release plan: Zenodo on acceptance).              |
| ArXiv-CS-Metadata  | arXiv CS-prefix metadata bulk download.                                    |
| Library-Sci-Meta   | Curated for this paper (release plan: Zenodo on acceptance).              |

## Empirical missingness rates

The realism-calibrated mask in `src/dltransclass/data/missingness.py` is anchored to these per-field observation rates (matching Table 2 of the paper):

| Dataset            | Title | Abstract | Keywords | Subject |
|--------------------|------:|---------:|---------:|--------:|
| UDC-OldBooks       | 1.00  | 0.72     | 0.41     | 0.41    |
| LCSH-Theses        | 1.00  | 0.83     | 0.38     | 0.38    |
| JUCS-Meta          | 1.00  | 0.91     | 0.77     | 0.77    |
| CompScholar-Meta   | 1.00  | 0.95     | 0.81     | 0.81    |
| ArXiv-CS-Metadata  | 1.00  | 0.98     | 0.92     | 0.92    |
| Library-Sci-Meta   | 1.00  | 0.81     | 0.45     | 0.45    |
