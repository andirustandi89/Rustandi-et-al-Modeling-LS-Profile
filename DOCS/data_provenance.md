# Data provenance

- **Main LMS cohort:** learner-level representation reconstructed from the public Moodle source associated with Riestra-González et al. [27] in the manuscript. The repository starts from the learner-level numeric and categorical matrices used by the reported experiments; the Moodle database dump is not required.
- **PENDIKOM questionnaire cohort:** `compiled_pendikom_moodle_logs.csv` from `https://github.com/Dewi2501/Learning-Style-Based/`. Only records with `has_ils = 1` are used for the independent ILS correspondence analysis.
- **Categorical benchmarks:** ten datasets from the UCI Machine Learning Repository. Archive checksums are retained in `DATA/BENCHMARKS/provenance_checksums.csv`.
