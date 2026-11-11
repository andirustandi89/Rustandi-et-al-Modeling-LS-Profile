"""Jalur terpadu. Hanya lokasi I/O; algoritme CARE-KM berada di care_km.py asli."""
from pathlib import Path
import os, json

GROUPS={'12_Sensitivitas_Aturan_Semantik': 'EKSPERIMEN/SEMANTIK/12_Sensitivitas_Aturan_Semantik', '09_Robustness_LMS_30x80': 'EKSPERIMEN/LMS/09_Robustness_LMS_30x80', '20_Statistik_Benchmark': 'EKSPERIMEN/GENERALISASI/20_Statistik_Benchmark', '14_Stabilitas_Partisi': 'EKSPERIMEN/SEMANTIK/14_Stabilitas_Partisi', '03_Lingkungan_dan_Reproduksibilitas': 'KONTROL/03_Lingkungan_dan_Reproduksibilitas', '17_Ablasi_Komponen_CARE_KM': 'EKSPERIMEN/LMS/17_Ablasi_Komponen_CARE_KM', '07_Eksperimen_LMS_Full': 'EKSPERIMEN/LMS/07_Eksperimen_LMS_Full', '06_Verifikasi_Inisialisasi_Baseline': 'EKSPERIMEN/VALIDASI/06_Verifikasi_Inisialisasi_Baseline', '21_Runtime_Stabilitas_Skalabilitas': 'EKSPERIMEN/GENERALISASI/21_Runtime_Stabilitas_Skalabilitas', '08_Lintasan_Refinement_CARE_KM': 'EKSPERIMEN/LMS/08_Lintasan_Refinement_CARE_KM', '11_Interpretasi_Semantik_FSLSM': 'EKSPERIMEN/SEMANTIK/11_Interpretasi_Semantik_FSLSM', '10_Statistik_LMS': 'EKSPERIMEN/LMS/10_Statistik_LMS', '22_Holdout_LMS_Pendikom': 'EKSPERIMEN/HOLDOUT', '19_Generalisasi_Benchmark': 'EKSPERIMEN/GENERALISASI/19_Generalisasi_Benchmark', '16_Sensitivitas_Candidate_Budget': 'EKSPERIMEN/LMS/16_Sensitivitas_Candidate_Budget', '23_Tabel_dan_Gambar': 'TABEL_DAN_GAMBAR/23_Tabel_dan_Gambar', '05_Verifikasi_Formulasi_CARE_KM': 'EKSPERIMEN/VALIDASI/05_Verifikasi_Formulasi_CARE_KM', '24_Gambar_Reviewer': 'TABEL_DAN_GAMBAR/24_Gambar_Reviewer', '18_Audit_Dataset_Benchmark': 'EKSPERIMEN/GENERALISASI/18_Audit_Dataset_Benchmark', '13_Stabilitas_Semantik': 'EKSPERIMEN/SEMANTIK/13_Stabilitas_Semantik', '15_Sensitivitas_Praproses': 'EKSPERIMEN/LMS/15_Sensitivitas_Praproses'}
LMS_FILES={
 'selected':'01_selected_userid_highest_activity.csv',
 'numeric':'02_dataset_behavioral_numeric.csv',
 'lmh':'03_dataset_behavioral_categorical_LMH.csv',
 'codes':'04_dataset_behavioral_categorical_fitur.csv',
 'categorical':'05_dataset_behavioral_categorical_skala.csv'}
PIPELINE_LAYOUT_VERSION='2026.11.06-experiments-3'

def find_root(root=None):
    explicit=root or os.environ.get('CARE_KM_ROOT') or os.environ.get('CARE_KM_PAPER_ROOT')
    p=Path(explicit) if explicit else Path(__file__).resolve().parent.parent
    p=p.expanduser().resolve()
    if not all((p/x).is_dir() for x in ('DATA','LIB','PYTHON')):
        raise FileNotFoundError(f'Root REVISED belum lengkap: {p}')
    return p

def configure(root):
    p=find_root(root)
    os.environ['CARE_KM_ROOT']=str(p);os.environ['CARE_KM_PAPER_ROOT']=str(p)
    for relative in set(GROUPS.values())|{'SPLIT/LMS/TRAIN','SPLIT/LMS/TEST',
            'SPLIT/PENDIKOM/TRAIN','SPLIT/PENDIKOM/TEST','SPLIT/RESAMPLING/LMS',
            'SPLIT/RESAMPLING/BENCHMARK','EKSPERIMEN/HOLDOUT/LMS','EKSPERIMEN/HOLDOUT/PENDIKOM','KONTROL'}:
        (p/'OUTPUT'/relative).mkdir(parents=True,exist_ok=True)
    return p

def stage_dir(stem,root=None):
    p=find_root(root)/'OUTPUT'/GROUPS[stem]
    p.mkdir(parents=True,exist_ok=True)
    return p

def lms_data(name='categorical',root=None):
    name=str(name)
    if name in LMS_FILES:filename=LMS_FILES[name]
    elif name in LMS_FILES.values():filename=name
    else:raise ValueError(f'Nama input LMS tidak dikenal: {name}. Gunakan nama hasil PRAPROSES final.')
    return find_root(root)/'OUTPUT/PRAPROSES/01_Praproses_main_LMS'/filename

def pendikom_data(kind,root=None):
    return find_root(root)/'OUTPUT/PRAPROSES/02_Praproses_Pendikom'/(
        '01_numeric_behavioral_pendikom.csv' if kind=='numeric' else '02_categorical_behavioral_pendikom.csv')

PREFIX={
 '00_':'03_Lingkungan_dan_Reproduksibilitas','02_':'05_Verifikasi_Formulasi_CARE_KM',
 '03_':'06_Verifikasi_Inisialisasi_Baseline','04_':'07_Eksperimen_LMS_Full',
 '05_':'08_Lintasan_Refinement_CARE_KM','06_':'09_Robustness_LMS_30x80','07_':'10_Statistik_LMS',
 '08_':'11_Interpretasi_Semantik_FSLSM','09_':'18_Audit_Dataset_Benchmark',
 '10_':'19_Generalisasi_Benchmark','11_':'20_Statistik_Benchmark',
 '12_':'21_Runtime_Stabilitas_Skalabilitas','13_':'23_Tabel_dan_Gambar',
 '18_':'16_Sensitivitas_Candidate_Budget','19_':'17_Ablasi_Komponen_CARE_KM',
 'semantic_rule_':'12_Sensitivitas_Aturan_Semantik','semantic_stability_':'13_Stabilitas_Semantik',
 'partition_stability_':'14_Stabilitas_Partisi','preprocessing_sensitivity_':'15_Sensitivitas_Praproses'}

def artifact_path(name,current=None,root=None):
    name=Path(name); first=name.parts[0]; p=find_root(root)
    if name.is_absolute() or '..' in name.parts:raise ValueError(name)
    if first.startswith('70_30_main_lms_'):
        if first in ('70_30_main_lms_split_indices.csv','70_30_main_lms_train_only_thresholds.csv'):
            return p/'OUTPUT/SPLIT/LMS'/name
        return p/'OUTPUT/EKSPERIMEN/HOLDOUT/LMS'/name
    if first.startswith('70_30_'):
        if first in ('70_30_groundtruth_split_audit.csv','70_30_groundtruth_feature_audit.csv',
                     '70_30_groundtruth_train_userids.csv','70_30_groundtruth_test_userids.csv'):
            return p/'OUTPUT/SPLIT/PENDIKOM'/name
        return p/'OUTPUT/EKSPERIMEN/HOLDOUT/PENDIKOM'/name
    if first=='FIGURES':return stage_dir('23_Tabel_dan_Gambar',p)/name
    if first=='FIGURES_REVIEWER':return stage_dir('24_Gambar_Reviewer',p)/name
    if first=='SEMANTIC_STABILITY_CHECKPOINTS':return stage_dir('13_Stabilitas_Semantik',p)/name
    if first=='18_Q_SENSITIVITY':return stage_dir('16_Sensitivitas_Candidate_Budget',p)/Path(*name.parts[1:])
    if first=='19_COMPONENT_ABLATION':return stage_dir('17_Ablasi_Komponen_CARE_KM',p)/Path(*name.parts[1:])
    owner=next((v for k,v in PREFIX.items() if first.startswith(k)),current)
    return (stage_dir(owner,p) if owner in GROUPS else p/'OUTPUT/KONTROL')/name

def protocol_path(name,root=None):return find_root(root)/'LIB/PROTOKOL'/name

def record_resample(stem,dataset,rep,indices,expected_hash=None):
    import hashlib
    import numpy as np
    import pandas as pd
    idx=np.asarray(indices,dtype=np.int64)
    h=hashlib.sha256(idx.tobytes()).hexdigest()[:16]
    if expected_hash is not None and h!=expected_hash:raise RuntimeError('Hash indeks resampling berubah')
    group='LMS' if dataset=='LMS' else 'BENCHMARK'
    folder=find_root()/'OUTPUT/SPLIT/RESAMPLING'/group/stem
    if dataset!='LMS':folder=folder/str(dataset)
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row_index':idx}).to_csv(folder/f'rep_{int(rep):02d}.csv',index=False)
