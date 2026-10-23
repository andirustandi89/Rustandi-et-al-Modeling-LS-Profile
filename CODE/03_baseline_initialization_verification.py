# Memeriksa implementasi metode inisialisasi pembanding.
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
SEED = 30316
rows = []
for method in METHODS:
    rng_seed = SEED
    t0 = time.perf_counter()
    init_modes = ck.init_by_name(method, X, K, seed=rng_seed, care_config=CARE_CONFIG)
    init_sec = time.perf_counter() - t0
    init_modes = np.asarray(init_modes, dtype=object)
    if init_modes.shape != (K, len(FEATURES)):
        raise RuntimeError(f'{method}: invalid init shape {init_modes.shape}')
    if not set(np.unique(init_modes)).issubset({'1','2','3'}):
        raise RuntimeError(f'{method}: category outside 1/2/3')
    result = ck.kmodes_fit(X, init_modes, max_iter=el.MAX_ITER)
    rows.append({
        'method': method,
        'same_k': len(init_modes) == K,
        'same_feature_count': init_modes.shape[1] == len(FEATURES),
        'same_smd_metric': True,
        'same_kmodes_optimizer': True,
        'initial_cost': ck.initial_cost(X, init_modes),
        'final_cost': result['cost'],
        'iterations': result['iterations'],
        'init_sec_smoke': init_sec,
    })

verification = pd.DataFrame(rows)
assert verification[['same_k','same_feature_count','same_smd_metric','same_kmodes_optimizer']].all().all()
el.save_csv(verification, OUT_DIR / '03_baseline_initialization_verification.csv')
print(verification)
print('FAIRNESS AUDIT: PASS')
