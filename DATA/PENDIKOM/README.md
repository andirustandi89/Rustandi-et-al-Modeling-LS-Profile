# PENDIKOM questionnaire cohort

The canonical source file is `compiled_pendikom_moodle_logs.csv` from the public `Dewi2501/Learning-Style-Based` repository. Run `python CODE/download_public_data.py --pendikom-only` to place the source file in this folder. The CARE-KM experiment retains `has_ils = 1`, giving 461 ILS-linked learners. `pendikom_ils_split_161600.csv` stores the frozen 322/139 train/test split. `pendikom_ils_461_numeric.csv` is included only as a compact audit artifact derived from the executed pipeline; it does not replace the canonical compiled source file.
