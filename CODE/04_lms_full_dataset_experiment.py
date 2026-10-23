# Menjalankan perbandingan clustering pada seluruh cohort LMS.
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
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '01_FULL_LMS'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el
import care_km as ck

CARE_CONFIG = dict(el.FROZEN_CARE_CONFIG)
K = el.K
METHODS = list(el.METHODS)
print('PAPER_ROOT:', PAPER_ROOT)
print('CARE config:', CARE_CONFIG)


df, X, FEATURES = el.load_lms_dataset(PAPER_ROOT)
STOCHASTIC_REPS = 15
DETERMINISTIC_REPS = 1
BASE_SEED = 4161600
CHECKPOINT = OUT_DIR / '04_full_dataset_raw.csv'

if CHECKPOINT.exists():
    raw = pd.read_csv(CHECKPOINT)
else:
    raw = pd.DataFrame()

for method in METHODS:
    reps = STOCHASTIC_REPS if method in el.STOCHASTIC_METHODS else DETERMINISTIC_REPS
    done = set()
    if len(raw):
        done = set(pd.to_numeric(raw.loc[raw['method'].astype(str)==method, 'rep'], errors='coerce').dropna().astype(int))

    for rep in range(reps):
        if rep in done:
            continue
        seed = BASE_SEED + rep
        print(f'run: {method:10s} rep={rep+1}/{reps}')
        row, _, _ = ck.run_method(
            method, X, None, K,
            seed=seed,
            care_config=CARE_CONFIG,
            si_sample_size=el.SI_SAMPLE_SIZE,
            si_seed=BASE_SEED + rep,
            max_iter=el.MAX_ITER,
        )
        row['rep'] = rep
        row['stochastic_initializer'] = method in el.STOCHASTIC_METHODS
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
        el.save_csv(raw, CHECKPOINT)


for method in METHODS:
    expected = STOCHASTIC_REPS if method in el.STOCHASTIC_METHODS else DETERMINISTIC_REPS
    actual = int((raw['method'].astype(str)==method).sum())
    if actual != expected:
        raise RuntimeError(f'{method}: raw runs {actual}, expected {expected}')

summary = el.summarize_runs(raw)
summary['rank_final_cost'] = summary['final_cost'].rank(method='average', ascending=True)
summary['rank_dbi'] = summary['dbi'].rank(method='average', ascending=True)
summary['rank_si'] = summary['si'].rank(method='average', ascending=False)
summary['rank_primary'] = summary[['rank_final_cost','rank_dbi','rank_si']].mean(axis=1)
summary = summary.sort_values(['rank_primary','final_cost'])
el.save_csv(summary, OUT_DIR / '04_full_dataset_summary.csv')


reference_seed = BASE_SEED
care_row, care_init_modes, care_result = ck.run_method(
    'CARE-KM', X, None, K,
    seed=reference_seed,
    care_config=CARE_CONFIG,
    si_sample_size=el.SI_SAMPLE_SIZE,
    si_seed=reference_seed,
    max_iter=el.MAX_ITER,
)
care_modes = pd.DataFrame(care_result['modes'], columns=FEATURES)
care_modes.insert(0, 'cluster_id', np.arange(K))
care_labels = pd.DataFrame({
    'user': df['user'].astype(str),
    'cluster_id': care_result['labels'].astype(int),
})
care_init_df = pd.DataFrame(care_init_modes, columns=FEATURES)
care_init_df.insert(0, 'mode_id', np.arange(K))
el.save_csv(care_init_df, OUT_DIR / '04_carekm_reference_initial_modes.csv')
el.save_csv(care_modes, OUT_DIR / '04_carekm_reference_final_modes.csv')
el.save_csv(care_labels, OUT_DIR / '04_carekm_reference_labels.csv')
(OUT_DIR / '04_carekm_reference.json').write_text(json.dumps({
    'seed': int(reference_seed),
    'initial_cost': float(care_row['initial_cost']),
    'final_cost': float(care_result['cost']),
    'dbi': float(care_row['dbi']),
    'si': float(care_row['si']),
    'iterations': int(care_result['iterations']),
    'parent_initializer': 'Cao',
    'care_config': CARE_CONFIG,
    'refinement_budget_rule': 'ceil(log2(candidate_budget))',
    'derived_refinement_budget': ck.derive_refinement_budget(CARE_CONFIG['candidate_budget']),
}, indent=2), encoding='utf-8')

print(summary)
print('CARE-KM reference seed:', reference_seed)
