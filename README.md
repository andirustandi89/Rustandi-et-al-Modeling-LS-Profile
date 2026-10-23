# Modeling Learning Style Profiles through Cluster-Derived Supervision

This repository contains the reproducibility package for the manuscript **“Modeling Learning Style Profiles through Cluster-Derived Supervision.”** It includes the CARE-KM implementation, local Python scripts, frozen learner splits, semantic mappings, benchmark metadata, and the outputs used in the manuscript.

## Repository layout

```text
DATA/
  MAIN_LMS/
  PENDIKOM/
  BENCHMARKS/
CODE/
  LIB/
  *.py
OUTPUT/
  01_FULL_LMS/
  02_ABLATION/
  03_Q_SENSITIVITY/
  04_ROBUSTNESS/
  05_CROSS_DATASET/
  06_FSLSM_PROFILES/
  07_STABILITY/
  08_MAIN_LMS_HOLDOUT/
  09_QUESTIONNAIRE_HOLDOUT/
  10_FIGURES/
SUPPLEMENTARY/
DOCS/
```

## Data

The main LMS experiment uses a 6,658-learner categorical matrix with 13 behavioral indicators. The learner-level numeric source used to construct that matrix is included under `DATA/MAIN_LMS/`. The raw Moodle database is not required by this repository.

The independent questionnaire experiment uses `compiled_pendikom_moodle_logs.csv` from the public [Learning-Style-Based](https://github.com/Dewi2501/Learning-Style-Based/) repository. The source CSV is fetched by `CODE/download_public_data.py` rather than duplicated in this archive. Records with `has_ils = 1` yield 461 ILS-linked learners. The frozen split contains 322 training learners and 139 held-out learners.

The cross-dataset evaluation uses ten UCI categorical benchmarks. Source archive checksums are stored under `DATA/BENCHMARKS/`.

## Local execution

```bash
python -m venv .venv
# Activate the environment using the command for your operating system
pip install -r requirements.txt
python CODE/download_public_data.py
python CODE/run_all.py
```

Validated manuscript outputs are already included. `run_all.py` skips a stage when its expected output is present. To recompute selected experiments:

```bash
python CODE/run_all.py --force --stage full-lms
python CODE/run_all.py --force --stage questionnaire-holdout
```

## Experimental constants

- Main LMS cluster count: `K = 16`
- K-Modes maximum iterations: `100`
- CARE-KM candidate budget: `q = 16`
- Derived refinement bound: `ceil(log2(q)) = 4`
- Holdout split seed: `161600`
- Main LMS holdout: 4,660 train / 1,998 test
- PENDIKOM ILS cohort: 322 train / 139 test
- LMS robustness: 30 matched 80% resamples
- Cross-dataset evaluation: 10 UCI datasets

## Manuscript outputs

See `DOCS/experiment_mapping.md` for the direct mapping between manuscript tables and repository scripts/output files. Canonical CSVs for the two holdout tables are `OUTPUT/08_MAIN_LMS_HOLDOUT/Table_11_Main_LMS_Classifier_Performance.csv` and `OUTPUT/09_QUESTIONNAIRE_HOLDOUT/Table_12_External_ILS_Correspondence.csv`.

## Data use

Third-party datasets remain subject to their original source licenses and terms. The code in this repository is provided under the repository license.
