# Reproducibility guide

1. Create a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Download the public PENDIKOM and UCI source files with `python CODE/download_public_data.py`.
4. Run `python CODE/run_all.py`. Existing validated outputs are skipped by default. Use `--force` to recompute a stage, or `--stage <name>` to run selected stages.

The scripts use paths relative to the repository root and do not require Google Colab, Google Drive, Jupyter, or a Moodle database dump.

The original executed output metadata records the research-source SHA-256 for `care_km.py`. The repository copy has a different file hash because comments, docstrings, and local path handling were cleaned for reviewer execution; the executable AST of the CARE-KM implementation is unchanged.

