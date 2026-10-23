# Memeriksa lingkungan, hash file, dan konfigurasi eksperimen.
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
print('CARE metadata:', el.frozen_care_metadata())


import platform
import sklearn
import scipy

DATASET_FILE = DATA_DIR / 'MAIN_LMS' / 'main_lms_categorical.csv'
CARE_FILE = LIB_DIR / 'care_km.py'


EXPECTED_CARE_KM_SHA256 = el.EXPECTED_CARE_KM_SHA256
EXPECTED_LMS_DATA_SHA256 = getattr(
    el,
    'EXPECTED_LMS_DATA_SHA256',
    '57fef508915e617faf6a2c4dfb1b72f6a00f6ec008b8622f703d016503a9b319'
)

if not CARE_FILE.is_file():
    raise FileNotFoundError(f'care_km.py tidak ditemukan: {CARE_FILE}')
if not DATASET_FILE.is_file():
    raise FileNotFoundError(f'dataset_categorical_skala.csv tidak ditemukan: {DATASET_FILE}')

care_hash = el.sha256_file(CARE_FILE)
if care_hash != EXPECTED_CARE_KM_SHA256:
    raise RuntimeError(
        'care_km.py hash mismatch. '
        f'expected={EXPECTED_CARE_KM_SHA256}, actual={care_hash}'
    )

dataset_hash = el.sha256_file(DATASET_FILE)
if dataset_hash != EXPECTED_LMS_DATA_SHA256:
    raise RuntimeError(
        'dataset_categorical_skala.csv hash mismatch. '
        f'expected={EXPECTED_LMS_DATA_SHA256}, actual={dataset_hash}'
    )

df, X, FEATURES = el.load_lms_dataset(PAPER_ROOT)
audit = el.dataset_audit(df, FEATURES)
el.save_csv(audit, OUT_DIR / '00_dataset_audit.csv')

protocol = {
    'paper_root': '.',
    'dataset_file': 'DATA/MAIN_LMS/main_lms_categorical.csv',
    'dataset_sha256': dataset_hash,
    'care_km_sha256': care_hash,
    'care_implementation_version': getattr(ck, 'CARE_IMPLEMENTATION_VERSION', 'unknown'),
    'K': K,
    'max_iter': el.MAX_ITER,
    'si_sample_size': el.SI_SAMPLE_SIZE,
    'methods': METHODS,
    'stochastic_methods': sorted(el.STOCHASTIC_METHODS),
    'care_config': CARE_CONFIG,
    'care_metadata': el.frozen_care_metadata(),
    'scale_treated_as_nominal_categories': True,
    'python': platform.python_version(),
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'sklearn': sklearn.__version__,
    'scipy': scipy.__version__,
}
(OUT_DIR / '00_protocol.json').write_text(
    json.dumps(protocol, indent=2),
    encoding='utf-8'
)

print(audit)
print(json.dumps(protocol, indent=2))
print('STATUS: PASS')
