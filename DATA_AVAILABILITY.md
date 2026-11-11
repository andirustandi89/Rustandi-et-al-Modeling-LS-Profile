# Data availability and preparation

This documentation package includes the processed datasets, split references, fitted states, predictions, model files, and recorded experiment outputs from the supplied archive. Raw inputs are supplied separately: the author will upload the original DATA files to GitHub, while the large `mysql-export` remains at the external URL below.

## Included artifacts

| Artifact | Location |
|---|---|
| LMS numerical and categorical learner matrices | `OUTPUT/PRAPROSES/01_Praproses_main_LMS/` |
| Pendikom numerical and categorical matrices | `OUTPUT/PRAPROSES/02_Praproses_Pendikom/` |
| Fixed learner split and ILS reference | `LIB/PROTOKOL/lms_split_indices.csv`, `LIB/PROTOKOL/pendikom_ils_labels.csv` |
| Train/test matrices and fitted LMS thresholds | `OUTPUT/SPLIT/LMS/`, `OUTPUT/SPLIT/PENDIKOM/` |
| Matched resampling indices | `OUTPUT/SPLIT/RESAMPLING/` |
| Pendikom retained-feature mask | `OUTPUT/SPLIT/PENDIKOM/70_30_groundtruth_feature_audit.csv` |
| LMS and Pendikom semantic rules | `OUTPUT/EKSPERIMEN/SEMANTIK/11_Interpretasi_Semantik_FSLSM/08_semantic_evidence_matrix.csv`, `OUTPUT/EKSPERIMEN/HOLDOUT/PENDIKOM/70_30_groundtruth_semantic_rules.csv` |
| Baseline configurations, modes, assignments, predictions, and model files | `OUTPUT/TAMBAHAN/25_*` through `OUTPUT/TAMBAHAN/32_*` |
| Canonical predictions and reference counts | `OUTPUT/TAMBAHAN/33_Analisis_Gabungan_Hasil/PREDIKSI/`, `EVALUASI/` |
| All-test/common-resolved analysis and original stage-34 source | `OUTPUT/TAMBAHAN/34_Analisis_Coverage_dan_Common_Resolved/` |

Processed rows are linked with the fixed learner identifiers and split metadata. Their inclusion supports auditing the published analysis; it is distinct from redistributing every upstream database. Source ownership and upstream dataset terms remain with the original providers. This documentation adds no blanket data license.

## Raw LMS export

Source context: Riestra-González, Paule-Ruíz, and Ortin, *Massive LMS Log Data Analysis for the Early Prediction of Course-Agnostic Student Performance* (2021), cited in the manuscript. Upstream project: <https://github.com/moisesriestra/moodle-early-performance-prediction>.

The author-provided download for the exact SQL input is:

<https://storage.googleapis.com/dissertation-data/dissertation-export/mysql-export>

Save it as `DATA/mysql-export`, without adding a `.txt` or `.html` suffix. The original run records a file size of **6,983,537,835 bytes** and SHA-256:

```text
53f88755e40e3243f656b72138e0a17299bd64986bf08b03706004bdab09b7f8
```

These are recorded fingerprints from `OUTPUT/KONTROL/status_pipeline.json`; the large remote file was not downloaded or independently rehashed during documentation. To download it on Linux or in a Colab shell:

```bash
mkdir -p DATA
curl --fail --location --retry 3 \
  --output DATA/mysql-export \
  'https://storage.googleapis.com/dissertation-data/dissertation-export/mysql-export'
```

Retain one active `mysql-export` input. The preprocessing resolver can accept supported SQL archives, but multiple competing candidates cause an error. A compressed/repacked file has a different byte checksum; compare the uncompressed `mysql-export` if checking the recorded fingerprint above. Stage 01 parses the SQL dump directly; it does not require importing the database into a running MySQL server.

## Raw Pendikom input

Place the original author-supplied compiled file at:

`DATA/compiled_pendikom_moodle_logs.csv`

Source project cited in the manuscript: <https://github.com/Dewi2501/Learning-Style-Based/>. The code consumes the compiled CSV, not an arbitrary replacement export. Its recorded size is **213,819 bytes**; the expected SHA-256 is in `DOCS/EXPECTED_DATA_FILES.json`.

The recorded source has 565 records, of which 461 have usable linked ILS data. The fixed holdout contains 322 train and 139 test learners. The pipeline begins with 19 candidate predictors and removes `navigation_skip` from the original holdout predictor set, retaining 18. Stages 02 and 32 read the raw compiled data. Stage 32 reconstructs its own course/temporal training contexts; the original 322/139 split is not substituted for those contexts.

## Benchmark inputs added manually to DATA

Source collection: [UCI Machine Learning Repository](https://archive.ics.uci.edu/), cited in the manuscript. Use the original archive copies used by the experiment, with the exact local filenames below. New downloads or repacked ZIPs may differ from the recorded byte fingerprints even if a dataset title is the same.

| Final Table 7 ID | Dataset | Expected local path |
|---|---|---|
| D1 | Car Evaluation | `DATA/dataset_car+evaluation.zip` |
| D2 | Lymphography | `DATA/dataset_lymphography.zip` |
| D3 | Hayes-Roth | `DATA/hayes+roth.zip` |
| D4 | Australian Credit (Statlog) | `DATA/statlog+australian+credit+approval.zip` |
| D5 | Chronic Kidney Disease | `DATA/chronic+kidney+disease.zip` |
| D6 | Annealing | `DATA/annealing.zip` |
| D7 | Statlog Heart | `DATA/statlog+heart.zip` |
| D8 | Heart Disease (Cleveland) | `DATA/heart+disease.zip` |
| D9 | New Thyroid | `DATA/thyroid+disease.zip` |
| D10 | Nursery | `DATA/dataset_nursery.zip` |

`LIB/generalization_data.py` defines archive-member selection, missing-value handling, categorical conversion, retained attributes, reference labels, and dataset order. For example, Annealing combines the supplied train/test files, Hayes-Roth drops the identifier/distractor fields specified by the loader, and the Thyroid loader uses the New Thyroid subset. Keep reference labels out of the feature matrix. The registry reference class count supplies benchmark k; it is not the LMS k=16.

CKD can use the loader's documented ARFF fallback if the original ZIP requires a RAR extractor. The matrix fingerprint checks still apply. Do not replace datasets or change discretization merely to make old output hashes match.

## Preparing and checking the completed DATA folder

1. Add the compiled Pendikom CSV and the ten original benchmark archives manually.
2. Download the SQL export from the external URL only when rebuilding LMS preprocessing.
3. From the project root, run `python DOCS/reproducibility.py check --require-data`.
4. Resolve any mismatch against the recorded source files. Then use `refresh-checksums` to include the manually added DATA files in a new release checksum inventory.

`DOCS/EXPECTED_DATA_FILES.json` lists the 12 active raw inputs, their recorded sizes/checksums, and their provenance. The same hashes are provided in standard two-space format in `DOCS/EXPECTED_DATA_SHA256SUMS.txt`. The distributed `SHA256SUMS.txt` covers only files actually included in the ZIP; it does not claim to hash absent raw data.

Legacy input inventories also mention other archives that are not active inputs to the documented 00–34 workflow. Those historical records are preserved, but they are not introduced here as additional dataset requirements. Adding/removing files in DATA can invalidate the existing global input cache, which inventories that directory; this is normal behavior of the original pipeline.

The external download URL is recorded as supplied by the author. Long-term availability of the remote server is not guaranteed by the package. Keep the recorded fingerprint with any independently archived copy of the SQL input.
