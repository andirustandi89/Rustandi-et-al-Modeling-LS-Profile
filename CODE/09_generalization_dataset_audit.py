# Mengaudit sepuluh dataset benchmark kategorikal.
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


DATASETS = gd.resolve_dataset_keys()
EXPECTED_DEFAULT = gd.DATASET_ORDER.copy()

if os.environ.get('CARE_KM_GEN_DATASETS', '').strip() == '':
    if DATASETS != EXPECTED_DEFAULT:
        raise RuntimeError('Urutan registry dataset generalization berubah.')

rows = []
for key in DATASETS:
    spec = gd.DATASETS[key]
    source = DATA_DIR / spec['file']
    if not source.exists():
        raise FileNotFoundError(f'{source} belum tersedia. Jalankan: python CODE/download_public_data.py --benchmarks-only')

    print(f'Audit: {key} <- {source.name}', flush=True)
    row = gd.audit_dataset(DATA_DIR, key)
    row['source_sha256'] = sha256_file(source)
    rows.append(row)

audit = pd.DataFrame(rows)

if audit['dataset'].duplicated().any():
    raise RuntimeError('Dataset duplikat pada audit.')
if not audit['valid'].astype(bool).all():
    bad = audit.loc[~audit['valid'].astype(bool), 'dataset'].tolist()
    raise RuntimeError(f'Dataset structurally invalid: {bad}')

audit.to_csv(OUT_DIR / '09_generalization_dataset_audit.csv', index=False)

source_hashes = {
    gd.DATASETS[k]['file']: sha256_file(DATA_DIR / gd.DATASETS[k]['file'])
    for k in DATASETS
}
provenance = {
    'source_location': 'DATA/BENCHMARKS',
    'dataset_count': int(len(DATASETS)),
    'dataset_keys': DATASETS,
    'methods': METHODS,
    'resampling_repetitions': 30,
    'resampling_fraction': 0.80,
    'matched_resampling': True,
    'without_replacement': True,
    'care_config': CARE_CONFIG,
    'care_metadata': el.frozen_care_metadata(),
    'source_files_sha256': source_hashes,
    'external_generalization_folder_used': False,
}
(OUT_DIR / '09_generalization_provenance.json').write_text(
    json.dumps(provenance, indent=2), encoding='utf-8'
)

print(audit)
warnings = audit.loc[~audit['metadata_match'].astype(bool), [
    'dataset','rows','features','classes',
    'reference_rows','reference_features','reference_k'
]]
if len(warnings):
    print('\nMetadata reference warning(s); structural audit tetap valid:')
    print(warnings)

print(json.dumps(provenance, indent=2))
print('STATUS: PASS')
