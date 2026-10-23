# Memeriksa formulasi dan mekanisme dasar CARE-KM.
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


X_toy = np.array([
    ['A','L','x'], ['A','L','y'], ['A','M','x'], ['A','M','y'],
    ['B','M','x'], ['B','M','z'], ['B','H','z'], ['B','H','y'],
    ['C','L','x'], ['C','L','z'], ['C','H','y'], ['C','H','z'],
], dtype=object)
K_TOY = 3


def smd_manual(a, b):
    return int(sum(x != y for x, y in zip(a, b)))

D_manual = np.array([[smd_manual(x, z) for z in X_toy[:K_TOY]] for x in X_toy])
D_lib = ck.smd_matrix(X_toy, X_toy[:K_TOY])
assert np.array_equal(D_manual, D_lib)
print('SMD direct verification: PASS')

cao_modes = ck.cao_init(X_toy, K_TOY)
cao_cost = ck.initial_cost(X_toy, cao_modes)
care_modes, trace, path = ck.care_init(
    X_toy, K_TOY, return_trace=True, return_path=True, **CARE_CONFIG
)
care_cost = ck.initial_cost(X_toy, care_modes)

trace_df = pd.DataFrame(trace)
derived_budget = ck.derive_refinement_budget(CARE_CONFIG['candidate_budget'])

assert derived_budget == int(np.ceil(np.log2(CARE_CONFIG['candidate_budget'])))
assert np.array_equal(path[0], cao_modes)
assert len(trace_df) <= derived_budget
if len(trace_df):
    assert (trace_df['gain'] > 0).all()
    assert (trace_df['cost_after'] < trace_df['cost_before']).all()
assert care_cost <= cao_cost + 1e-12


X_enc = ck._encode_categorical_for_care(X_toy)
D_enc = ck._smd_matrix_encoded(X_enc, X_enc[:K_TOY])
assert np.array_equal(D_lib, D_enc)

stopping_reason = (
    'NO_POSITIVE_GAIN'
    if len(trace_df) < derived_budget
    else 'DERIVED_REFINEMENT_BUDGET_REACHED'
)

verification = pd.DataFrame([{
    'toy_rows': len(X_toy),
    'toy_features': X_toy.shape[1],
    'k': K_TOY,
    'parent_initializer': 'Cao',
    'candidate_budget': CARE_CONFIG['candidate_budget'],
    'derived_refinement_budget': derived_budget,
    'cao_initial_cost': cao_cost,
    'care_initial_cost': care_cost,
    'accepted_exchanges': len(trace_df),
    'stopping_reason': stopping_reason,
    'smd_direct_equals_library': True,
    'encoded_smd_equals_object_smd': True,
    'path_starts_from_cao': True,
    'care_nonworsening_vs_cao': bool(care_cost <= cao_cost + 1e-12),
}])

el.save_csv(verification, OUT_DIR / '02_formulation_verification.csv')
el.save_csv(trace_df, OUT_DIR / '02_exchange_trace.csv')
el.save_csv(pd.DataFrame(care_modes), OUT_DIR / '02_toy_care_initial_modes.csv')
print(verification)
print(trace_df)
print('STATUS: PASS')
