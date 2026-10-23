# Menjalankan stage eksperimen secara berurutan dan melewati output yang sudah tersedia.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'CODE'

STAGES = [
    ('prepare-main', '01_prepare_main_lms.py', ROOT/'DATA/MAIN_LMS/main_lms_categorical.csv'),
    ('environment', '00_environment_and_reproducibility.py', ROOT/'OUTPUT/01_FULL_LMS/00_protocol.json'),
    ('formulation', '02_care_km_formulation_verification.py', ROOT/'OUTPUT/01_FULL_LMS/02_formulation_verification.csv'),
    ('baseline', '03_baseline_initialization_verification.py', ROOT/'OUTPUT/01_FULL_LMS/03_baseline_initialization_verification.csv'),
    ('full-lms', '04_lms_full_dataset_experiment.py', ROOT/'OUTPUT/01_FULL_LMS/04_full_dataset_summary.csv'),
    ('robustness', '06_lms_robustness_30x80.py', ROOT/'OUTPUT/04_ROBUSTNESS/06_robustness_summary.csv'),
    ('semantic', '08_fslsm_semantic_interpretation.py', ROOT/'OUTPUT/06_FSLSM_PROFILES/08_cluster_to_fslsm_mapping.csv'),
    ('ablation', '19_care_km_component_ablation.py', ROOT/'OUTPUT/02_ABLATION/19_component_ablation_table.csv'),
    ('q-sensitivity', '18_candidate_budget_sensitivity.py', ROOT/'OUTPUT/03_Q_SENSITIVITY/18_q_sensitivity_summary.csv'),
    ('benchmark-audit', '09_generalization_dataset_audit.py', ROOT/'OUTPUT/05_CROSS_DATASET/09_generalization_dataset_audit.csv'),
    ('cross-dataset', '10_cross_dataset_generalization.py', ROOT/'OUTPUT/05_CROSS_DATASET/10_average_rank.csv'),
    ('cross-stats', '11_cross_dataset_statistical_analysis.py', ROOT/'OUTPUT/05_CROSS_DATASET/11_friedman_kendall.csv'),
    ('runtime', '12_runtime_stability_and_scalability.py', ROOT/'OUTPUT/05_CROSS_DATASET/12_stability_failure_method_summary.csv'),
    ('semantic-stability', '21_semantic_stability.py', ROOT/'OUTPUT/07_STABILITY/semantic_stability_dimension_summary.csv'),
    ('partition-stability', '20_partition_stability.py', ROOT/'OUTPUT/07_STABILITY/partition_stability_summary.csv'),
    ('semantic-rule', '22_semantic_rule_sensitivity.py', ROOT/'OUTPUT/07_STABILITY/semantic_rule_sensitivity_summary.csv'),
    ('preprocessing-sensitivity', '23_preprocessing_sensitivity.py', ROOT/'OUTPUT/07_STABILITY/preprocessing_sensitivity_summary.csv'),
    ('main-holdout', '24_main_lms_holdout.py', ROOT/'OUTPUT/08_MAIN_LMS_HOLDOUT/Table_11_Main_LMS_Classifier_Performance_numeric.csv'),
    ('questionnaire-holdout', '17_pendikom_questionnaire_validation.py', ROOT/'OUTPUT/09_QUESTIONNAIRE_HOLDOUT/Table_12_External_ILS_Correspondence_numeric.csv'),
    ('artifacts', '13_generate_manuscript_artifacts.py', ROOT/'SUPPLEMENTARY/Table_S1_Full_FSLSM_Profiles.csv'),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Jalankan ulang stage meskipun output sudah ada.')
    parser.add_argument('--stage', action='append', help='Nama stage yang ingin dijalankan. Bisa diulang.')
    args = parser.parse_args()
    selected = set(args.stage or [])
    for name, script, marker in STAGES:
        if selected and name not in selected:
            continue
        if marker.exists() and not args.force:
            print(f'[lewati] {name}: output sudah ada')
            continue
        print(f'[jalan] {name}')
        subprocess.run([sys.executable, str(CODE/script)], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
