# Menguji sensitivitas candidate budget q tanpa membuat laporan naratif.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import json
import os
import sys
import time
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
DATA_DIR = PAPER_ROOT / 'DATA'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '03_Q_SENSITIVITY'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el
import care_km as ck

K = el.K
MAX_ITER = el.MAX_ITER
SI_SAMPLE_SIZE = el.SI_SAMPLE_SIZE

print('PAPER_ROOT:', PAPER_ROOT)
print('CARE implementation:', getattr(ck, 'CARE_IMPLEMENTATION_VERSION', 'unknown'))
print('K:', K, '| MAX_ITER:', MAX_ITER, '| SI sample size:', SI_SAMPLE_SIZE)
print('Frozen CARE configuration:', el.FROZEN_CARE_CONFIG)


Q_VALUES = [2, 4, 8, 16, 32, 64]
REPS = 20
FRACTION = 0.80
BASE_SEED = 5161600

CANDIDATE_RULE = str(el.FROZEN_CARE_CONFIG['candidate_rule'])
MIN_GAIN = float(el.FROZEN_CARE_CONFIG['min_gain'])


PRIMARY_Q = 16
assert PRIMARY_Q in Q_VALUES
EXPECTED_BUDGETS = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6}
for q, expected in EXPECTED_BUDGETS.items():
    actual = int(ck.derive_refinement_budget(q))
    assert actual == expected, f'q={q}: expected E_max={expected}, got {actual}'

RAW_PATH = OUT_DIR / '18_q_sensitivity_raw.csv'
SUMMARY_PATH = OUT_DIR / '18_q_sensitivity_summary.csv'
PAIRED_PATH = OUT_DIR / '18_q_sensitivity_paired_vs_q16.csv'
PROTOCOL_PATH = OUT_DIR / '18_q_sensitivity_protocol.json'


protocol = {
    'purpose': 'diagnostic sensitivity analysis; not hyperparameter tuning or model selection',
    'q_values': Q_VALUES,
    'primary_q': PRIMARY_Q,
    'refinement_budget_rule': 'ceil(log2(candidate_budget))',
    'derived_budgets': {str(q): int(ck.derive_refinement_budget(q)) for q in Q_VALUES},
    'repetitions': REPS,
    'fraction': FRACTION,
    'base_seed': BASE_SEED,
    'matched_resamples': True,
    'K': int(K),
    'candidate_rule': CANDIDATE_RULE,
    'min_gain': MIN_GAIN,
    'max_kmodes_iterations': int(MAX_ITER),
    'si_sample_size': int(SI_SAMPLE_SIZE),
    'care_implementation_version': getattr(ck, 'CARE_IMPLEMENTATION_VERSION', 'unknown'),
    'frozen_q16_metadata': el.frozen_care_metadata(),
}
PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
print(json.dumps(protocol, indent=2))


df, X_all, FEATURES = el.load_lms_dataset(PAPER_ROOT)

dataset_path = DATA_DIR / 'MAIN_LMS' / 'main_lms_categorical.csv'
dataset_sha256 = el.sha256_file(dataset_path)
care_source_path = LIB_DIR / 'care_km.py'
care_sha256 = el.sha256_file(care_source_path)

print('Dataset:', dataset_path)
print('Rows:', len(df), '| Features:', len(FEATURES))
print('Dataset SHA-256:', dataset_sha256)
print('care_km.py SHA-256:', care_sha256)
print('Expected CARE SHA-256:', getattr(el, 'EXPECTED_CARE_KM_SHA256', 'n/a'))

audit = el.dataset_audit(df, FEATURES)
print(audit)

if len(df) != 6658:
    raise RuntimeError(f'Expected 6658 learners, got {len(df)}')
if len(FEATURES) != 13:
    raise RuntimeError(f'Expected 13 features, got {len(FEATURES)}')


protocol.update({
    'dataset_path': 'DATA/MAIN_LMS/main_lms_categorical.csv',
    'dataset_sha256': dataset_sha256,
    'care_source_path': 'CODE/LIB/care_km.py',
    'care_source_sha256': care_sha256,
})
PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2), encoding='utf-8')


if RAW_PATH.exists():
    raw = pd.read_csv(RAW_PATH)
else:
    raw = pd.DataFrame()

expected_pairs = {(rep, q) for rep in range(REPS) for q in Q_VALUES}
if len(raw):
    completed_pairs = set(
        zip(
            pd.to_numeric(raw['rep'], errors='coerce').astype('Int64'),
            pd.to_numeric(raw['q'], errors='coerce').astype('Int64'),
        )
    )
else:
    completed_pairs = set()

for rep in range(REPS):
    rng = np.random.default_rng(BASE_SEED + rep)
    idx = np.sort(
        rng.choice(
            len(X_all),
            size=int(len(X_all) * FRACTION),
            replace=False,
        )
    )
    X = X_all[idx]
    sample_hash = el.sample_index_hash(idx)

    print(f'\nrep={rep+1:02d}/{REPS} | n={len(X)} | sample_hash={sample_hash}')

    for q in Q_VALUES:
        if (rep, q) in completed_pairs:
            print(f'  q={q}: SKIP checkpoint')
            continue

        care_config = {
            'candidate_rule': CANDIDATE_RULE,
            'candidate_budget': int(q),
            'min_gain': MIN_GAIN,
        }
        derived_budget = int(ck.derive_refinement_budget(q))

        t0 = time.perf_counter()
        init_modes, trace = ck.care_init(
            X,
            K,
            return_trace=True,
            **care_config,
        )
        t1 = time.perf_counter()

        fit = ck.kmodes_fit(X, init_modes, max_iter=MAX_ITER)
        t2 = time.perf_counter()

        metrics = ck.evaluate_clustering(
            X,
            None,
            fit,
            init_modes,
            si_sample_size=SI_SAMPLE_SIZE,
            si_seed=BASE_SEED + rep,
        )
        t3 = time.perf_counter()

        accepted = int(len(trace))
        stopping_reason = (
            'NO_POSITIVE_GAIN'
            if accepted < derived_budget
            else 'DERIVED_REFINEMENT_BUDGET_REACHED'
        )

        row = {
            'rep': rep,
            'rep_display': rep + 1,
            'q': int(q),
            'derived_refinement_budget': derived_budget,
            'accepted_corrections': accepted,
            'stopping_reason': stopping_reason,
            'sample_n': int(len(X)),
            'sample_hash': sample_hash,
            **metrics,
            'care_init_sec': float(t1 - t0),
            'kmodes_sec': float(t2 - t1),
            'evaluation_sec': float(t3 - t2),
            'algorithm_sec': float(t2 - t0),
            'total_sec': float(t3 - t0),
        }

        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
        raw = raw.sort_values(['rep','q']).reset_index(drop=True)
        el.save_csv(raw, RAW_PATH)
        completed_pairs.add((rep, q))

        print(
            f"  q={q:2d} | Emax={derived_budget} | accepted={accepted} | "
            f"init={metrics['initial_cost']:.1f} | final={metrics['final_cost']:.1f} | "
            f"DBI={metrics['dbi']:.4f} | SI={metrics['si']:.4f} | "
            f"iter={metrics['iterations']} | alg_sec={row['algorithm_sec']:.2f}"
        )


actual_pairs = set(zip(raw['rep'].astype(int), raw['q'].astype(int)))
missing = sorted(expected_pairs - actual_pairs)
if missing:
    raise RuntimeError(f'Checkpoint belum lengkap. Missing rep/q pairs: {missing[:10]}')

if raw.duplicated(['rep','q']).any():
    raise RuntimeError('Duplicate rep/q rows found.')


if not (raw.groupby('rep')['sample_hash'].nunique() == 1).all():
    raise RuntimeError('Matched-resample integrity FAILED: sample hashes differ across q.')

print('\nMatched-resample integrity: PASS')
print('Complete rows:', len(raw), 'expected:', REPS * len(Q_VALUES))
print(raw.head(9))

summary = (
    raw.groupby(['q','derived_refinement_budget'], as_index=False)
    .agg(
        n=('rep','size'),
        accepted_corrections_mean=('accepted_corrections','mean'),
        accepted_corrections_sd=('accepted_corrections','std'),
        initial_cost_mean=('initial_cost','mean'),
        initial_cost_sd=('initial_cost','std'),
        final_cost_mean=('final_cost','mean'),
        final_cost_sd=('final_cost','std'),
        dbi_mean=('dbi','mean'),
        dbi_sd=('dbi','std'),
        si_mean=('si','mean'),
        si_sd=('si','std'),
        iterations_mean=('iterations','mean'),
        iterations_sd=('iterations','std'),
        care_init_sec_mean=('care_init_sec','mean'),
        algorithm_sec_mean=('algorithm_sec','mean'),
        total_sec_mean=('total_sec','mean'),
    )
    .sort_values('q')
    .reset_index(drop=True)
)


early = (
    raw.assign(early_stop=raw['stopping_reason'].eq('NO_POSITIVE_GAIN'))
    .groupby('q')['early_stop']
    .agg(['sum','mean'])
    .reset_index()
    .rename(columns={'sum':'early_stop_n','mean':'early_stop_rate'})
)
summary = summary.merge(early, on='q', how='left')

el.save_csv(summary, SUMMARY_PATH)
print(summary)
print('Saved:', SUMMARY_PATH)

METRIC_DIRECTIONS = {
    'initial_cost': 'lower',
    'final_cost': 'lower',
    'dbi': 'lower',
    'si': 'higher',
    'iterations': 'lower',
    'algorithm_sec': 'lower',
}

rows = []
for metric, direction in METRIC_DIRECTIONS.items():
    base = raw.loc[raw['q'].eq(PRIMARY_Q), ['rep', metric]].set_index('rep')[metric].sort_index()
    metric_rows = []

    for q_cmp in [q for q in Q_VALUES if q != PRIMARY_Q]:
        comp = raw.loc[raw['q'].eq(q_cmp), ['rep', metric]].set_index('rep')[metric].sort_index()
        common = base.index.intersection(comp.index)
        a = base.loc[common].to_numpy(dtype=float)
        b = comp.loc[common].to_numpy(dtype=float)
        diff = a - b

        if direction == 'lower':
            wins = int(np.sum(a < b))
            losses = int(np.sum(a > b))
        else:
            wins = int(np.sum(a > b))
            losses = int(np.sum(a < b))
        ties = int(np.sum(np.isclose(a, b, rtol=0.0, atol=1e-12)))


        if np.all(np.isclose(diff, 0.0, atol=1e-12, rtol=0.0)):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')

        metric_rows.append({
            'metric': metric,
            'direction': direction,
            'primary_q': PRIMARY_Q,
            'comparator_q': q_cmp,
            'n_pairs': len(common),
            'q16_mean': float(np.mean(a)),
            'comparator_mean': float(np.mean(b)),
            'mean_difference_q16_minus_comparator': float(np.mean(diff)),
            'median_difference_q16_minus_comparator': float(np.median(diff)),
            'q16_wins': wins,
            'ties': ties,
            'q16_losses': losses,
            'wilcoxon_statistic': float(stat),
            'wilcoxon_p_two_sided': float(p),
        })

    adj = el.holm_adjust([r['wilcoxon_p_two_sided'] for r in metric_rows])
    for r, p_adj in zip(metric_rows, adj):
        r['holm_p_within_metric'] = float(p_adj)
        rows.append(r)

paired = pd.DataFrame(rows)
el.save_csv(paired, PAIRED_PATH)
print(paired)
print('Saved:', PAIRED_PATH)


main_metrics = ['initial_cost','final_cost','dbi','si','iterations','algorithm_sec']
stability_rows = []
summary_by_q = summary.set_index('q')

for metric in main_metrics:
    col = f'{metric}_mean'
    if col not in summary_by_q.columns:
        continue

    means = summary_by_q[col]
    vals = means.loc[Q_VALUES].to_numpy(dtype=float)
    denom = abs(float(means.loc[PRIMARY_Q]))
    relative_range_pct = (
        float(vals.max() - vals.min()) / denom * 100.0
        if denom > 0 else np.nan
    )

    row = {
        'metric': metric,
        'q16_mean': float(means.loc[PRIMARY_Q]),
        'min_mean_across_q': float(vals.min()),
        'max_mean_across_q': float(vals.max()),
        'range_across_q': float(vals.max() - vals.min()),
        'range_relative_to_q16_pct': relative_range_pct,
    }
    for q in Q_VALUES:
        row[f'q{q}_mean'] = float(means.loc[q])
    stability_rows.append(row)

stability = pd.DataFrame(stability_rows)
print(stability)
