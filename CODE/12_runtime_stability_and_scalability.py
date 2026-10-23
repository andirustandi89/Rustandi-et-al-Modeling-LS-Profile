# Meringkas runtime, skalabilitas, dan audit kegagalan.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import json
import os
import sys
import hashlib

import numpy as np
import pandas as pd


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
DATA_DIR = PAPER_ROOT / 'DATA' / 'BENCHMARKS'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '05_CROSS_DATASET'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

for _name in ['experiment_lib', 'care_km', 'generalization_data']:
    sys.modules.pop(_name, None)
import experiment_lib as el
import care_km as ck
import generalization_data as gd


METHODS = list(el.METHODS)
CARE_CONFIG = dict(el.FROZEN_CARE_CONFIG)

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

print('PAPER_ROOT:', PAPER_ROOT)
print('DATA_DIR  :', DATA_DIR)
print('METHODS   :', METHODS)
print('CARE config:', CARE_CONFIG)


audit_file = OUT_DIR / '09_generalization_dataset_audit.csv'
full_file = OUT_DIR / '10_cross_dataset_full.csv'
full_raw_file = OUT_DIR / '10_cross_dataset_full_raw.csv'
res_file = OUT_DIR / '10_cross_dataset_resample_summary.csv'
res_raw_file = OUT_DIR / '10_cross_dataset_resample_raw.csv'

for p in [audit_file, full_file, full_raw_file, res_file, res_raw_file]:
    if not p.exists():
        raise FileNotFoundError(p)

audit = pd.read_csv(audit_file)
full = pd.read_csv(full_file)
full_raw = pd.read_csv(full_raw_file)
res = pd.read_csv(res_file)
res_raw = pd.read_csv(res_raw_file)

DATASETS = audit['dataset'].astype(str).tolist()

for name, frame in [
    ('full', full), ('full_raw', full_raw),
    ('resample', res), ('resample_raw', res_raw)
]:
    actual = set(frame['dataset'].astype(str).unique())
    if actual != set(DATASETS):
        raise RuntimeError(f'{name}: daftar dataset tidak sinkron dengan stage 09.')

meta = audit[['dataset','rows','features','classes']].copy()
meta['n_times_d'] = meta['rows'] * meta['features']
meta['n_times_d_times_k'] = meta['rows'] * meta['features'] * meta['classes']

full_runtime = full[
    ['dataset','method','init_sec','cluster_sec','total_sec','iterations']
].copy().rename(columns={
    'init_sec':'init_sec_mean',
    'cluster_sec':'cluster_sec_mean',
    'total_sec':'total_sec_mean',
    'iterations':'iterations_mean',
})

res_runtime = res[
    ['dataset','method',
     'init_sec_mean','init_sec_sd',
     'cluster_sec_mean','cluster_sec_sd',
     'total_sec_mean','total_sec_sd',
     'iterations_mean','iterations_sd']
].copy()

full_runtime = full_runtime.merge(meta, on='dataset', how='left')
res_runtime = res_runtime.merge(meta, on='dataset', how='left')

def add_ratio_vs_cao(df):
    out = df.copy()
    cao = (
        out[out['method']=='Cao'][['dataset','init_sec_mean']]
        .rename(columns={'init_sec_mean':'cao_init_sec'})
    )
    out = out.merge(cao, on='dataset', how='left')
    out['init_ratio_vs_cao'] = (
        out['init_sec_mean'] / out['cao_init_sec'].replace(0, np.nan)
    )
    return out

full_runtime = add_ratio_vs_cao(full_runtime)
res_runtime = add_ratio_vs_cao(res_runtime)

full_runtime.to_csv(OUT_DIR / '12_full_runtime_scalability.csv', index=False)
res_runtime.to_csv(OUT_DIR / '12_resample_runtime_scalability.csv', index=False)

care_runtime = res_runtime[res_runtime['method']=='CARE-KM'].copy().sort_values('rows')
care_runtime.to_csv(OUT_DIR / '12_care_runtime_scalability.csv', index=False)

corr_rows = []
for size_col in ['rows','features','classes','n_times_d','n_times_d_times_k']:
    for runtime_col in ['init_sec_mean','cluster_sec_mean','total_sec_mean']:
        pair = care_runtime[[size_col,runtime_col]].dropna()
        rho = pair.corr(method='spearman').iloc[0,1] if len(pair) >= 2 else np.nan
        corr_rows.append({
            'size_variable': size_col,
            'runtime_variable': runtime_col,
            'spearman_rho': float(rho) if pd.notna(rho) else np.nan,
            'n_datasets': int(len(pair)),
        })

corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT_DIR / '12_care_scalability_correlations.csv', index=False)

print(care_runtime)
print(corr)


full_raw = full_raw.copy()
full_raw['source'] = 'full'
res_raw = res_raw.copy()
res_raw['source'] = 'resample'

required = [
    'initial_cost','final_cost','dbi','si','iterations',
    'min_cluster','max_cluster','cluster_cv',
    'init_sec','cluster_sec','total_sec','method','dataset'
]
for name, frame in [('full_raw', full_raw), ('resample_raw', res_raw)]:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f'{name}: kolom stability hilang: {missing}')

raw = pd.concat(
    [full_raw[required + ['source']], res_raw[required + ['source']]],
    ignore_index=True
)

numeric_core = [
    'initial_cost','final_cost','dbi','si','iterations',
    'min_cluster','max_cluster','cluster_cv',
    'init_sec','cluster_sec','total_sec'
]
arr = raw[numeric_core].apply(pd.to_numeric, errors='coerce')
raw['invalid_numeric'] = ~np.isfinite(arr.to_numpy(dtype=float)).all(axis=1)
raw['empty_cluster'] = pd.to_numeric(
    raw['min_cluster'], errors='coerce'
).fillna(0) <= 0
raw['objective_worsened'] = (
    pd.to_numeric(raw['final_cost'], errors='coerce')
    > pd.to_numeric(raw['initial_cost'], errors='coerce') + 1e-12
)
raw['nonpositive_iterations'] = (
    pd.to_numeric(raw['iterations'], errors='coerce').fillna(0) <= 0
)
raw['cluster_cv_gt_1'] = (
    pd.to_numeric(raw['cluster_cv'], errors='coerce') > 1.0
)
raw['algorithmic_failure'] = (
    raw['invalid_numeric']
    | raw['empty_cluster']
    | raw['objective_worsened']
    | raw['nonpositive_iterations']
)

raw.to_csv(OUT_DIR / '12_stability_failure_raw_flags.csv', index=False)

summary = (
    raw.groupby(['source','dataset','method'], as_index=False)
    .agg(
        runs=('method','size'),
        invalid_runs=('invalid_numeric','sum'),
        empty_cluster_runs=('empty_cluster','sum'),
        objective_worsened_runs=('objective_worsened','sum'),
        nonpositive_iteration_runs=('nonpositive_iterations','sum'),
        algorithmic_failure_runs=('algorithmic_failure','sum'),
        cluster_cv_gt_1_runs=('cluster_cv_gt_1','sum'),
        min_cluster_min=('min_cluster','min'),
        min_cluster_median=('min_cluster','median'),
        max_cluster_max=('max_cluster','max'),
        cluster_cv_mean=('cluster_cv','mean'),
        cluster_cv_sd=('cluster_cv','std'),
        iterations_mean=('iterations','mean'),
        iterations_sd=('iterations','std'),
    )
)
summary['success_runs'] = summary['runs'] - summary['algorithmic_failure_runs']
summary['success_rate'] = summary['success_runs'] / summary['runs']
summary['failure_rate'] = summary['algorithmic_failure_runs'] / summary['runs']
summary.to_csv(OUT_DIR / '12_stability_failure_summary.csv', index=False)

method_summary = (
    summary.groupby(['source','method'], as_index=False)
    .agg(
        total_runs=('runs','sum'),
        total_failures=('algorithmic_failure_runs','sum'),
        datasets_with_failure=(
            'algorithmic_failure_runs',
            lambda s: int((s > 0).sum())
        ),
        mean_cluster_cv=('cluster_cv_mean','mean'),
        mean_success_rate=('success_rate','mean'),
    )
)
method_summary.to_csv(
    OUT_DIR / '12_stability_failure_method_summary.csv',
    index=False
)

print(method_summary)
print(f'STATUS: PASS | {len(DATASETS)} datasets')
