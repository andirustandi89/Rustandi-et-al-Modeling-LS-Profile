# Mengukur kestabilan partition dengan ARI, NMI, dan AMI.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT/'CODE'/'LIB'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '07_STABILITY'
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el

MEMBER_FILE = OUT_DIR/'semantic_stability_member_comparison.csv'
REFERENCE_FILE = PAPER_ROOT/'OUTPUT'/'01_FULL_LMS'/'04_carekm_reference_labels.csv'
ROBUST_FILE = PAPER_ROOT/'OUTPUT'/'04_ROBUSTNESS'/'06_robustness_raw.csv'

if not MEMBER_FILE.is_file():
    raise FileNotFoundError(
        'semantic_stability_member_comparison.csv tidak ditemukan. '
        'Jalankan semantic_stability_FIXED.py terlebih dahulu.'
    )
if not REFERENCE_FILE.is_file():
    raise FileNotFoundError(
        '04_carekm_reference_labels.csv tidak ditemukan.'
    )

member = pd.read_csv(
    MEMBER_FILE,
    dtype={'user':str,'sample_hash':str},
)
reference = pd.read_csv(
    REFERENCE_FILE,
    dtype={'user':str},
)

required = {'rep','sample_hash','user','cluster_id_resample'}
if not required.issubset(member.columns):
    raise RuntimeError(
        f'Kolom member comparison tidak lengkap: {sorted(required-set(member.columns))}'
    )

member['rep'] = pd.to_numeric(member['rep'], errors='raise').astype(int)
member['cluster_id_resample'] = pd.to_numeric(
    member['cluster_id_resample'], errors='raise'
).astype(int)
reference['cluster_id'] = pd.to_numeric(
    reference['cluster_id'], errors='raise'
).astype(int)

if reference['user'].duplicated().any():
    raise RuntimeError('Reference labels memiliki user duplikat.')

ref_map = reference.set_index('user')['cluster_id']
member['cluster_id_reference'] = member['user'].map(ref_map)

if member['cluster_id_reference'].isna().any():
    raise RuntimeError('Ada user resample yang tidak ditemukan pada full-data reference.')

member['cluster_id_reference'] = member['cluster_id_reference'].astype(int)

if member['rep'].nunique() != 30:
    raise RuntimeError(
        f'Diharapkan 30 resamples, ditemukan {member["rep"].nunique()}.'
    )

print('PAPER_ROOT:', PAPER_ROOT)
print('Member-resample rows:', len(member))
print('Repetitions:', member['rep'].nunique())


hash_check = []

if ROBUST_FILE.is_file():
    robust = pd.read_csv(ROBUST_FILE, dtype={'sample_hash':str})
    robust['rep'] = pd.to_numeric(robust['rep'], errors='raise').astype(int)

    robust_hash = (
        robust.groupby('rep')['sample_hash']
        .agg(lambda x: sorted(set(x.astype(str))))
    )

    if not robust_hash.apply(len).eq(1).all():
        raise RuntimeError('06_lms_robustness_30x80.py memiliki lebih dari satu sample_hash pada rep yang sama.')

    robust_hash = robust_hash.apply(lambda x: x[0])

    for rep in sorted(member['rep'].unique()):
        observed = sorted(
            member.loc[member['rep'].eq(rep),'sample_hash']
            .astype(str).unique()
        )
        if len(observed) != 1:
            raise RuntimeError(f'Rep {rep} memiliki lebih dari satu sample_hash.')

        expected = str(robust_hash.loc[rep])
        hash_check.append({
            'rep':int(rep),
            'semantic_stability_hash':observed[0],
            'robustness_notebook06_hash':expected,
            'match':observed[0] == expected,
        })

    hash_check = pd.DataFrame(hash_check)
    if not hash_check['match'].all():
        raise RuntimeError('Sample hash tidak identik dengan 06_lms_robustness_30x80.py.')

    print('Sample-hash verification against 06_lms_robustness_30x80.py: PASS')
else:
    hash_check = pd.DataFrame(columns=[
        'rep','semantic_stability_hash',
        'robustness_notebook06_hash','match'
    ])
    print('06_robustness_raw.csv tidak ditemukan; hash cross-check dilewati.')

rows = []

for rep, g in member.groupby('rep', sort=True):
    y_ref = g['cluster_id_reference'].to_numpy(dtype=int)
    y_res = g['cluster_id_resample'].to_numpy(dtype=int)

    rows.append({
        'rep':int(rep),
        'sample_hash':str(g['sample_hash'].iloc[0]),
        'sample_n':int(len(g)),
        'reference_clusters_present':int(pd.Series(y_ref).nunique()),
        'resample_clusters_present':int(pd.Series(y_res).nunique()),
        'ari':float(adjusted_rand_score(y_ref, y_res)),
        'nmi':float(normalized_mutual_info_score(y_ref, y_res)),
        'ami':float(adjusted_mutual_info_score(y_ref, y_res)),
    })

metrics = pd.DataFrame(rows).sort_values('rep').reset_index(drop=True)

if len(metrics) != 30:
    raise RuntimeError('Metrics tidak memiliki tepat 30 repetitions.')

el.save_csv(
    metrics,
    OUT_DIR/'partition_stability_resample_metrics.csv',
    index=False
)

print(metrics)

N_BOOT = 10000
BOOT_SEED = 20261014

def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    boot = rng.choice(
        values,
        size=(N_BOOT, len(values)),
        replace=True
    ).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025,0.975])
    return float(lo), float(hi)


summary_rows = []

for i, metric in enumerate(['ari','nmi','ami']):
    values = metrics[metric].to_numpy(dtype=float)
    lo, hi = bootstrap_mean_ci(values, BOOT_SEED+i)

    summary_rows.append({
        'metric':metric.upper(),
        'n_reps':int(len(values)),
        'mean':float(values.mean()),
        'sd':float(values.std(ddof=1)),
        'median':float(np.median(values)),
        'min':float(values.min()),
        'max':float(values.max()),
        'bootstrap_95ci_low':lo,
        'bootstrap_95ci_high':hi,
    })

summary = pd.DataFrame(summary_rows)

el.save_csv(
    summary,
    OUT_DIR/'partition_stability_summary.csv',
    index=False
)

if len(hash_check):
    el.save_csv(
        hash_check,
        OUT_DIR/'partition_stability_sample_integrity.csv',
        index=False
    )

print(summary)

validation = {
    'analysis':'CARE-KM partition stability across 30 matched 80% resamples',
    'reference_partition':'04_carekm_reference_labels.csv',
    'resample_partition_source':'semantic_stability_member_comparison.csv',
    'repetitions':30,
    'label_alignment_required':False,
    'hungarian_assignment_used':False,
    'metrics':['ARI','NMI','AMI'],
    'sample_hash_verified_against_notebook_06':bool(
        len(hash_check) and hash_check['match'].all()
    ),
    'outputs':[
        'partition_stability_resample_metrics.csv',
        'partition_stability_summary.csv',
        'partition_stability_sample_integrity.csv',
    ],
}

(OUT_DIR/'partition_stability_validation.json').write_text(
    json.dumps(validation, indent=2), encoding='utf-8'
)

print(json.dumps(validation, indent=2))
