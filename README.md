# Behavior Driven Cluster Derived Supervision for Learning Style Profiling

This package contains the experiment notebooks, shared Python modules, saved predictions, models, and analysis outputs for the study. The main LMS cohort has 6,658 learners and 13 categorical indicators. The ILS-linked Pendikom cohort has 461 learners, with a 322/139 train/test split and 18 retained predictors.

The package covers master notebook 00, notebooks 01–33, and the existing Python analysis for stage 34. Master 00 runs stages 01–24. Stages 25–33 are opened separately; stage 34 has its own entry point.

## Package layout

| Location | Contents |
|---|---|
| `DATA/` | Original inputs added by the author; the large MySQL export is downloaded separately |
| `LIB/` | CARE-KM, baseline helpers, dataset loaders, path handling, split references, and cache logic |
| `PYTHON/` | The 34 notebooks numbered 00–33 and the original `requirements.txt` |
| `OUTPUT/PRAPROSES/` | Numerical/categorical datasets and preprocessing audits |
| `OUTPUT/SPLIT/` | Fixed learner splits, resampling indices, train-fitted thresholds, and feature masks |
| `OUTPUT/EKSPERIMEN/` | Main-pipeline experiment results |
| `OUTPUT/TAMBAHAN/` | Extended experiments and analyses 25–34, including predictions and models |
| `TABLE_MANIFEST.csv` | Crosswalk for final manuscript Tables 1–23 |
| `DOCS/` | Complete execution map, expected input hashes, manuscript table reference, and verification tools |
| `SHA256SUMS.txt` | SHA-256 hashes of the files distributed in this package, excluding the checksum file itself |

All relative paths in the documentation start at the directory containing this README. For Google Colab, copy the package contents into `MyDrive/REVISED/`, so `LIB/`, `PYTHON/`, `DATA/`, and `OUTPUT/` are direct children of that directory. Keep the existing filenames and output subdirectories.

## Preparing the data

The exported ZIP does not contain raw DATA inputs. The author will add the compiled Pendikom CSV and the ten benchmark archives to `DATA/` on GitHub. Download `mysql-export` from the author-provided URL and save it as `DATA/mysql-export` before rebuilding LMS preprocessing:

<https://storage.googleapis.com/dissertation-data/dissertation-export/mysql-export>

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the exact filenames, sources, preparation steps, expected hashes, and the difference between included processed artifacts and external raw inputs. Existing result tables can be inspected without downloading the raw SQL file.

## Preparing the environment

Use a Python environment with NumPy, pandas, SciPy, scikit-learn, matplotlib, IPython, joblib, and threadpoolctl. The original pipeline dependencies remain in `PYTHON/requirements.txt`:

```bash
python -m pip install -r PYTHON/requirements.txt
```

In Colab, mount Drive, set the working directory to `/content/drive/MyDrive/REVISED`, and install from that path. Restart the runtime after changing scientific package versions, then open the required notebook and run its cells in order.

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/REVISED')
```

The saved later experiments record a different environment from the legacy requirements file. For example, stage 25 records Python 3.13.18, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3, and scikit-learn 1.6.1. Stage 31 also records matplotlib 3.10.0. These are recorded experiment versions, not a claim that the legacy requirements file recreates every saved environment. The per-stage protocol JSON files are authoritative for cache/model reuse. A different environment may require recomputing upstream results; never bypass a source or version check to reuse an incompatible model.

The CKD source archive may contain a RAR file. The existing loader supports several extractors, or an extracted `chronic_kidney_disease_full.arff`/`chronic_kidney_disease.arff` placed in `DATA/`. Legacy compressed `.Z` dataset members require `gzip`. No data loader or scientific dependency setting was changed by the documentation update.

## Running stages 00–34

The complete stage-by-stage reading order, dependencies, script locations, and outputs are in [DOCS/PIPELINE_00_34.md](DOCS/PIPELINE_00_34.md).

### Main pipeline

Open `PYTHON/00_Jalankan_Pipeline_REVISED.ipynb`. It validates inputs and executes notebooks 01–24 in the established order. Its distributed settings are:

```python
JALANKAN_PRAPROSES = False
JALANKAN_BENCHMARK = True
LANJUTKAN = True
```

Use `JALANKAN_PRAPROSES=True` to rebuild raw preprocessing; this requires both raw LMS and compiled Pendikom inputs. With `False`, the included processed datasets are used. Benchmark stages still require the original ten benchmark archives. `LANJUTKAN=True` reuses only checkpoints that pass the existing input, code, environment, and output checks. Invalidated checkpoints can be archived and their stages recomputed by the existing pipeline.

### Extended experiments

After the main inputs/results are available, open notebooks 25–33 separately. The numerical reading order is 25, 26, 27, 28, 29, 30, 31, 32, 33. Dependency details matter: 29 does not require 28; 30 requires 25–28, not 29; 31 requires 25 and the raw benchmark archives; 32 reconstructs new training contexts and requires the raw compiled Pendikom CSV. Stage 33 expects completed full-protocol results from 25–32.

Keep the complete `OUTPUT` tree, including `KONTROL`, `BANK`, `MODELS`, predictions, and bootstrap banks. Later analyses verify these files. Do not move stage 34 into `PYTHON/` as an extra notebook: the existing master inventory registers only the original notebooks and the explicitly supported stages 25–33.

### Stage 34

Stage 34 is the existing script:

`OUTPUT/TAMBAHAN/34_Analisis_Coverage_dan_Common_Resolved/PROTOKOL/analysis34_source.py`

It reads the completed stage-33 predictions and stored bootstrap weights from stages 27 and 32. It performs no clustering, model fitting, new prediction, or new random draw. The documentation helper calls its existing `run` function with its original default configuration:

```bash
python DOCS/reproducibility.py run-stage34
```

Run this from the project root. In Colab, execute the command in a code cell after changing to the project directory. Outputs are written into the existing stage-34 output directory. This is an explicit analysis action; it refreshes stage-34 results using the existing source.

## Locating final manuscript tables

`TABLE_MANIFEST.csv` follows Tables 1–23 in `MANUSCRIPT_ALUR_KOLOM_DIPERBAIKI.docx`, the latest manuscript available for this package. Its SHA-256 and the extracted table reference are recorded in `DOCS/manuscript_tables.json`. If the manuscript changes, update the manifest against that new document before freezing a public release.

The numeric prefixes in historical output filenames are preserved. For example, final Table 16 reads the historical `T12_Holdout_LMS_per_dimensi.csv`; final Table 20 comes from stage 34; final Table 22 comes directly from stage 29. The manifest gives the exact scripts, input/output paths, filtering, units, and display transformations. Path-list cells contain JSON arrays.

There are two documented provenance cautions in [DOCS/TABLE_SOURCE_NOTES.md](DOCS/TABLE_SOURCE_NOTES.md): the manuscript timing column in Table 6 does not match the active stored summary, and the old synthesis `S04_Sensitivitas_k.csv` is not the canonical source for final Table 22. Source numbers and scientific code were preserved; these issues are not hidden by relabeling outputs.

## Integrity checks

To verify the distributed files and the manifest paths:

```bash
python DOCS/reproducibility.py check
```

Missing raw DATA inputs are reported as pending because they are supplied separately. After adding the original DATA files, verify their recorded source hashes as well:

```bash
python DOCS/reproducibility.py check --require-data
```

To include manually added DATA files or authorized documentation edits in a new checksum inventory:

```bash
python DOCS/reproducibility.py refresh-checksums
python DOCS/reproducibility.py check --require-data
```

Refreshing checksums records the files currently present; it does not establish that altered scientific results reproduce the manuscript. The independent expected DATA fingerprints remain in `DOCS/EXPECTED_DATA_FILES.json`. Keep the public commit SHA and release version with the final manuscript citation. Freeze the release after the manual DATA upload and final checks.

## Scope of the cleanup and verification

Repeated notebook introductions and editorial prose were removed or shortened. Scientific code cells, the eight original Python files, configurations, seeds, saved models, and research outputs were preserved. Technical comments and docstrings remain where they describe behavior or are included in source fingerprints. Detailed script inventory, byte comparisons, syntax checks, and functional verification are in `DOCS/VERIFICATION_REPORT.md` and `DOCS/VERIFICATION.json`.

A full raw-data rerun in Google Colab was not performed during this packaging task. The raw DATA files were not downloaded, including the approximately 7 GB SQL export. Recorded source hashes and the new package checksum inventory serve different purposes and are labeled separately.
