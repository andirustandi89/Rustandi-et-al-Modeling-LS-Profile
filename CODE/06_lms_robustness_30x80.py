# Menjalankan 30 matched resamples pada cohort LMS.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import json
import os
import sys
import time
import hashlib

import numpy as np
import pandas as pd


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
DATA_DIR = PAPER_ROOT / 'DATA'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '04_ROBUSTNESS'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el
import care_km as ck

CARE_CONFIG = dict(el.FROZEN_CARE_CONFIG)
K = el.K
METHODS = list(el.METHODS)
print('PAPER_ROOT:', PAPER_ROOT)
print('CARE config:', CARE_CONFIG)


df, X_all, FEATURES = el.load_lms_dataset(PAPER_ROOT)
REPS = 30
FRACTION = 0.80
BASE_SEED = 6161600
CHECKPOINT = OUT_DIR / '06_robustness_raw.csv'

if CHECKPOINT.exists():
    raw = pd.read_csv(CHECKPOINT)
else:
    raw = pd.DataFrame()

for rep in range(REPS):
    rng = np.random.default_rng(BASE_SEED + rep)
    idx = np.sort(rng.choice(len(X_all), size=int(len(X_all)*FRACTION), replace=False))
    X = X_all[idx]
    sample_hash = el.sample_index_hash(idx)

    done = set()
    if len(raw):
        done = set(raw.loc[pd.to_numeric(raw['rep']) == rep, 'method'].astype(str))

    new_rows = []
    for method in METHODS:
        if method in done:
            continue
        row, _, _ = ck.run_method(
            method, X, None, K,
            seed=BASE_SEED + rep,
            care_config=CARE_CONFIG,
            si_sample_size=el.SI_SAMPLE_SIZE,
            si_seed=BASE_SEED + rep,
            max_iter=el.MAX_ITER,
        )
        row.update({'rep': rep, 'sample_hash': sample_hash, 'sample_n': len(idx)})
        new_rows.append(row)

    if new_rows:
        raw = pd.concat([raw, pd.DataFrame(new_rows)], ignore_index=True)
        el.save_csv(raw, CHECKPOINT)


counts = raw.groupby('rep')['method'].nunique()
if len(counts) != REPS or not (counts == len(METHODS)).all():
    raise RuntimeError('Robustness checkpoint belum lengkap 30 x 7.')
if not (raw.groupby('rep')['sample_hash'].nunique() == 1).all():
    raise RuntimeError('Matched-sample integrity gagal.')

summary = raw.groupby('method').agg(
    n=('rep','size'),
    initial_cost_mean=('initial_cost','mean'), initial_cost_sd=('initial_cost','std'),
    final_cost_mean=('final_cost','mean'), final_cost_sd=('final_cost','std'),
    dbi_mean=('dbi','mean'), dbi_sd=('dbi','std'),
    si_mean=('si','mean'), si_sd=('si','std'),
    init_sec_mean=('init_sec','mean'), init_sec_sd=('init_sec','std'),
    cluster_sec_mean=('cluster_sec','mean'), cluster_sec_sd=('cluster_sec','std'),
    total_sec_mean=('total_sec','mean'), total_sec_sd=('total_sec','std'),
    iterations_mean=('iterations','mean'), iterations_sd=('iterations','std'),
).reset_index()
summary['rank_final_cost'] = summary['final_cost_mean'].rank(ascending=True)
summary['rank_dbi'] = summary['dbi_mean'].rank(ascending=True)
summary['rank_si'] = summary['si_mean'].rank(ascending=False)
summary['rank_primary'] = summary[['rank_final_cost','rank_dbi','rank_si']].mean(axis=1)
summary = summary.sort_values(['rank_primary','final_cost_mean'])

el.save_csv(summary, OUT_DIR / '06_robustness_summary.csv')
print(summary)
print('Matched resampling integrity: PASS')
