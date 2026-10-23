# Menghitung Friedman, Kendall, Wilcoxon, dan koreksi Holm.
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


from scipy.stats import friedmanchisquare, wilcoxon

source_file = OUT_DIR / '10_cross_dataset_resample_summary.csv'
audit_file = OUT_DIR / '09_generalization_dataset_audit.csv'
if not source_file.exists():
    raise FileNotFoundError('Jalankan 10_cross_dataset_generalization.py terlebih dahulu.')
if not audit_file.exists():
    raise FileNotFoundError('Jalankan 09_generalization_dataset_audit.py terlebih dahulu.')

summary = pd.read_csv(source_file)
audit = pd.read_csv(audit_file)
DATASETS = audit['dataset'].astype(str).tolist()

if set(DATASETS) != set(summary['dataset'].astype(str).unique()):
    raise RuntimeError('Daftar dataset statistik tidak sinkron dengan audit.')
if len(DATASETS) < 3:
    raise RuntimeError('Friedman membutuhkan sedikitnya tiga dataset/blok yang lengkap.')

METRICS = {
    'dbi_mean': 'lower',
    'si_mean': 'higher',
    'final_cost_mean': 'lower',
    'initial_cost_mean': 'lower',
    'iterations_mean': 'lower',
    'total_sec_mean': 'lower',
}

def holm_adjust(p_values):
    return el.holm_adjust(p_values)

def rank_biserial_signed(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank(method='average').to_numpy()
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    return float((pos-neg)/(pos+neg)) if (pos+neg) else 0.0

friedman_rows = []
pair_rows = []

for metric, direction in METRICS.items():
    pivot = (
        summary.pivot(index='dataset', columns='method', values=metric)
        .reindex(index=DATASETS, columns=METHODS)
    )
    if pivot.isna().any().any():
        raise RuntimeError(f'Missing values pada {metric}.')

    samples = [pivot[m].to_numpy(dtype=float) for m in METHODS]
    stat, p = friedmanchisquare(*samples)
    n, m = pivot.shape
    friedman_rows.append({
        'metric': metric,
        'friedman_chi2': float(stat),
        'p_value': float(p),
        'kendall_w': float(stat/(n*(m-1))),
        'n_datasets': int(n),
        'n_methods': int(m),
    })

    care = pivot['CARE-KM']
    family = []
    for baseline in [x for x in METHODS if x != 'CARE-KM']:
        ref = pivot[baseline]
        improvement = (ref-care) if direction == 'lower' else (care-ref)
        nz = improvement[np.abs(improvement) > 1e-12]

        if len(nz) == 0:
            wstat, p_pair = 0.0, 1.0
        else:
            try:
                wstat, p_pair = wilcoxon(
                    improvement,
                    zero_method='wilcox',
                    alternative='two-sided',
                    method='auto',
                )
                wstat, p_pair = float(wstat), float(p_pair)
            except ValueError:
                wstat, p_pair = np.nan, 1.0

        family.append({
            'metric': metric,
            'direction': direction,
            'baseline': baseline,
            'n_datasets': int(len(improvement)),
            'wins': int((improvement > 1e-12).sum()),
            'ties': int((np.abs(improvement) <= 1e-12).sum()),
            'losses': int((improvement < -1e-12).sum()),
            'mean_improvement': float(improvement.mean()),
            'median_improvement': float(np.median(improvement)),
            'wilcoxon_stat': wstat,
            'p_value': p_pair,
            'rank_biserial': rank_biserial_signed(improvement.to_numpy()),
        })

    adjusted = holm_adjust([x['p_value'] for x in family])
    for row, p_adj in zip(family, adjusted):
        row['holm_p'] = float(p_adj)
        row['significant_0_05'] = bool(p_adj < 0.05)
        pair_rows.append(row)

friedman = pd.DataFrame(friedman_rows)
pairwise = pd.DataFrame(pair_rows)

friedman.to_csv(OUT_DIR / '11_friedman_kendall.csv', index=False)
pairwise.to_csv(OUT_DIR / '11_cross_dataset_pairwise_wilcoxon_holm.csv', index=False)

print(friedman)
print(pairwise)
print(f'STATUS: PASS | {len(DATASETS)} datasets')
