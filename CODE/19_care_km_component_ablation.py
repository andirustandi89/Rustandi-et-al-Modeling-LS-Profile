# Menjalankan component ablation CARE-KM pada matched resamples.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
DATA_DIR = PAPER_ROOT / 'DATA'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '02_ABLATION'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el
import care_km as ck

K = int(el.K)
MAX_ITER = int(el.MAX_ITER)
SI_SAMPLE_SIZE = int(el.SI_SAMPLE_SIZE)
CARE_CONFIG = dict(el.FROZEN_CARE_CONFIG)

print('PAPER_ROOT:', PAPER_ROOT)
print('CARE implementation:', getattr(ck, 'CARE_IMPLEMENTATION_VERSION', 'unknown'))
print('Frozen CARE config:', CARE_CONFIG)
print('Frozen CARE metadata:', el.frozen_care_metadata())


REPS = 20
FRACTION = 0.80
BASE_SEED = 5161600

Q = int(CARE_CONFIG['candidate_budget'])
MIN_GAIN = float(CARE_CONFIG['min_gain'])
DERIVED_BUDGET = int(ck.derive_refinement_budget(Q))

assert Q == 16, f'Expected frozen q=16, got {Q}'
assert DERIVED_BUDGET == 4, f'Expected derived E_max=4, got {DERIVED_BUDGET}'
assert str(CARE_CONFIG['candidate_rule']) == 'separation'

VARIANT_SPECS = [
    {
        'variant_id': 'A0',
        'variant': 'A0 Cao parent',
        'use_refinement': False,
        'use_regions': None,
        'screening_rule': None,
        'is_full_care': False,
    },
    {
        'variant_id': 'A1',
        'variant': 'A1 Global + density',
        'use_refinement': True,
        'use_regions': False,
        'screening_rule': 'density',
        'is_full_care': False,
    },
    {
        'variant_id': 'A2',
        'variant': 'A2 Region + density',
        'use_refinement': True,
        'use_regions': True,
        'screening_rule': 'density',
        'is_full_care': False,
    },
    {
        'variant_id': 'A3',
        'variant': 'A3 Global + separation',
        'use_refinement': True,
        'use_regions': False,
        'screening_rule': 'separation',
        'is_full_care': False,
    },
    {
        'variant_id': 'A4',
        'variant': 'A4 Full CARE-KM',
        'use_refinement': True,
        'use_regions': True,
        'screening_rule': 'separation',
        'is_full_care': True,
    },
]
VARIANT_ORDER = [v['variant_id'] for v in VARIANT_SPECS]
VARIANT_MAP = {v['variant_id']: v for v in VARIANT_SPECS}

RAW_PATH = OUT_DIR / '19_component_ablation_raw.csv'
SUMMARY_PATH = OUT_DIR / '19_component_ablation_summary.csv'
FULL_PAIRWISE_PATH = OUT_DIR / '19_component_ablation_pairwise_vs_full.csv'
CONTRAST_PATH = OUT_DIR / '19_component_ablation_factorial_contrasts.csv'
TABLE_PATH = OUT_DIR / '19_component_ablation_table.csv'
PROTOCOL_PATH = OUT_DIR / '19_component_ablation_protocol.json'


print('Variants:')
print(pd.DataFrame(VARIANT_SPECS))
print('q =', Q, '| E_max =', DERIVED_BUDGET)


def _ablation_refine(
    X,
    k,
    *,
    use_regions,
    screening_rule,
    candidate_budget=16,
    min_gain=0.0,
    return_trace=False,
):

    X = np.asarray(X, dtype=object)
    n = len(X)
    q = int(candidate_budget)
    refinement_budget = int(ck.derive_refinement_budget(q))

    X_encoded = ck._encode_categorical_for_care(X)
    density = ck._density_cao_encoded(X_encoded)
    mode_idx = ck._cao_indices_encoded(X_encoded, k, density)
    d_current = ck._smd_matrix_encoded(X_encoded, X_encoded[mode_idx])

    labels = np.argmin(d_current, axis=1)
    current_cost = float(d_current[np.arange(n), labels].sum())
    trace = []
    stopping_reason = 'DERIVED_REFINEMENT_BUDGET_REACHED'
    all_idx = np.arange(n, dtype=np.int64)

    for step in range(refinement_budget):
        labels = np.argmin(d_current, axis=1)
        d_first = d_current[np.arange(n), labels]

        if k > 1:
            d_second = np.partition(d_current, 1, axis=1)[:, 1]
        else:
            d_second = np.full(n, np.inf)

        best_gain = float(min_gain)
        best = None
        best_distance_column = None

        for h in range(k):
            eligible = np.flatnonzero(labels == h) if use_regions else all_idx
            if len(eligible) == 0:
                continue

            if screening_rule == 'density':
                score = density[eligible].astype(float)

            elif screening_rule == 'separation':
                if use_regions:

                    score = d_second[eligible].astype(float)
                else:


                    if k > 1:
                        other_cols = np.concatenate(
                            [d_current[:, :h], d_current[:, h+1:]],
                            axis=1,
                        )
                        competitive_sep = np.min(other_cols, axis=1)
                    else:
                        competitive_sep = np.ones(n, dtype=float)
                    score = competitive_sep[eligible].astype(float)
            else:
                raise ValueError(f'Unknown screening_rule={screening_rule}')

            order = np.argsort(-score, kind='stable')
            cand_idx = eligible[order[: min(q, len(eligible))]]

            if k > 1:
                best_other = np.where(labels == h, d_second, d_first)
            else:
                best_other = np.full(n, np.inf)

            candidate_encoded = X_encoded[cand_idx]
            candidate_dist = ck._smd_matrix_encoded(X_encoded, candidate_encoded)
            trial_costs = np.minimum(
                best_other[:, None],
                candidate_dist,
            ).sum(axis=0).astype(float)

            for pos, ci in enumerate(cand_idx):
                if np.array_equal(X_encoded[ci], X_encoded[mode_idx[h]]):
                    continue

                trial_cost = float(trial_costs[pos])
                gain = current_cost - trial_cost

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (h, int(ci), trial_cost)
                    best_distance_column = candidate_dist[:, pos].copy()

        if best is None:
            stopping_reason = 'NO_POSITIVE_GAIN'
            break

        h, row_id, new_cost = best
        mode_idx = mode_idx.copy()
        mode_idx[h] = row_id
        d_current[:, h] = best_distance_column

        trace.append({
            'step': step + 1,
            'cluster': int(h),
            'row': int(row_id),
            'gain': float(best_gain),
            'cost_before': float(current_cost),
            'cost_after': float(new_cost),
            'candidate_budget': q,
            'derived_refinement_budget': refinement_budget,
            'use_regions': bool(use_regions),
            'screening_rule': screening_rule,
        })
        current_cost = new_cost

    modes = X[mode_idx].copy()
    if return_trace:
        return modes, trace, stopping_reason
    return modes


def _official_full_care(X):
    modes, trace = ck.care_init(
        X,
        K,
        candidate_budget=Q,
        candidate_rule='separation',
        min_gain=MIN_GAIN,
        return_trace=True,
    )
    stopping_reason = (
        'NO_POSITIVE_GAIN'
        if len(trace) < DERIVED_BUDGET
        else 'DERIVED_REFINEMENT_BUDGET_REACHED'
    )
    return modes, trace, stopping_reason


def _cao_parent(X):
    return ck.cao_init(X, K), [], 'NO_REFINEMENT_PARENT'


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
print(el.dataset_audit(df, FEATURES))

if len(df) != 6658:
    raise RuntimeError(f'Expected 6658 learners, got {len(df)}')
if len(FEATURES) != 13:
    raise RuntimeError(f'Expected 13 features, got {len(FEATURES)}')
if getattr(el, 'EXPECTED_CARE_KM_SHA256', None) not in (None, '', care_sha256):
    raise RuntimeError('care_km.py SHA-256 differs from frozen expected source.')

protocol = {
    'purpose': 'true CARE-KM component ablation; diagnostic, not model selection',
    'dataset_path': 'DATA/MAIN_LMS/main_lms_categorical.csv',
    'dataset_sha256': dataset_sha256,
    'care_source_path': 'CODE/LIB/care_km.py',
    'care_source_sha256': care_sha256,
    'care_implementation_version': getattr(ck, 'CARE_IMPLEMENTATION_VERSION', 'unknown'),
    'repetitions': REPS,
    'fraction': FRACTION,
    'base_seed': BASE_SEED,
    'matched_resamples': True,
    'K': K,
    'candidate_budget': Q,
    'refinement_budget_rule': 'ceil(log2(candidate_budget))',
    'derived_refinement_budget': DERIVED_BUDGET,
    'min_gain': MIN_GAIN,
    'max_kmodes_iterations': MAX_ITER,
    'si_sample_size': SI_SAMPLE_SIZE,
    'variants': VARIANT_SPECS,
    'planned_contrasts': [
        'A4 vs A0: Full CARE-KM vs Cao parent',
        'A2 vs A1: region effect under density screening',
        'A4 vs A3: region effect under separation screening',
        'A3 vs A1: separation effect without region restriction',
        'A4 vs A2: separation effect with region restriction',
    ],
    'statistical_test': 'two-sided paired Wilcoxon signed-rank',
    'multiple_testing': 'Holm correction across five planned contrasts within each metric',
}
PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
print('Protocol saved:', PROTOCOL_PATH)


rng = np.random.default_rng(BASE_SEED)
idx0 = np.sort(rng.choice(len(X_all), size=int(len(X_all) * FRACTION), replace=False))
X0 = X_all[idx0]

official_modes, official_trace, _ = _official_full_care(X0)
local_modes, local_trace, _ = _ablation_refine(
    X0,
    K,
    use_regions=True,
    screening_rule='separation',
    candidate_budget=Q,
    min_gain=MIN_GAIN,
    return_trace=True,
)

if not np.array_equal(official_modes, local_modes):
    raise RuntimeError('PARITY FAIL: local region+separation modes differ from official CARE-KM.')
if len(official_trace) != len(local_trace):
    raise RuntimeError('PARITY FAIL: accepted-correction counts differ.')

for a, b in zip(official_trace, local_trace):
    if int(a['cluster']) != int(b['cluster']) or int(a['row']) != int(b['row']):
        raise RuntimeError('PARITY FAIL: accepted replacement differs.')
    if abs(float(a['cost_after']) - float(b['cost_after'])) > 1e-9:
        raise RuntimeError('PARITY FAIL: accepted cost differs.')

print('Official/full local parity: PASS')
print('Accepted corrections:', len(official_trace))


if RAW_PATH.exists():
    raw = pd.read_csv(RAW_PATH)
else:
    raw = pd.DataFrame()

expected_pairs = {(rep, vid) for rep in range(REPS) for vid in VARIANT_ORDER}

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

    shift = rep % len(VARIANT_ORDER)
    run_order = VARIANT_ORDER[shift:] + VARIANT_ORDER[:shift]

    for order_pos, variant_id in enumerate(run_order):
        spec = VARIANT_MAP[variant_id]

        done = False
        if len(raw):
            m = (
                pd.to_numeric(raw['rep'], errors='coerce').eq(rep)
                & raw['variant_id'].astype(str).eq(variant_id)
            )
            done = bool(m.any())

        if done:
            print(f'rep={rep+1:02d}/{REPS} {variant_id}: SKIP checkpoint')
            continue

        print(f'rep={rep+1:02d}/{REPS} {variant_id}: {spec["variant"]}')

        t0 = time.perf_counter()

        if variant_id == 'A0':
            init_modes, trace, stopping_reason = _cao_parent(X)
        elif variant_id == 'A4':
            init_modes, trace, stopping_reason = _official_full_care(X)
        else:
            init_modes, trace, stopping_reason = _ablation_refine(
                X,
                K,
                use_regions=bool(spec['use_regions']),
                screening_rule=str(spec['screening_rule']),
                candidate_budget=Q,
                min_gain=MIN_GAIN,
                return_trace=True,
            )

        init_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        result = ck.kmodes_fit(X, init_modes, max_iter=MAX_ITER)
        cluster_sec = time.perf_counter() - t1
        total_sec = init_sec + cluster_sec

        row = ck.evaluate_clustering(
            X,
            None,
            result,
            init_modes,
            si_sample_size=SI_SAMPLE_SIZE,
            si_seed=BASE_SEED + rep,
        )

        row.update({
            'rep': rep,
            'variant_id': variant_id,
            'variant': spec['variant'],
            'variant_order': VARIANT_ORDER.index(variant_id),
            'execution_order_within_rep': order_pos,
            'use_refinement': bool(spec['use_refinement']),
            'use_regions': spec['use_regions'],
            'screening_rule': spec['screening_rule'],
            'is_full_care': bool(spec['is_full_care']),
            'candidate_budget': Q,
            'derived_refinement_budget': DERIVED_BUDGET,
            'accepted_corrections': len(trace),
            'early_stop': bool(len(trace) < DERIVED_BUDGET) if spec['use_refinement'] else False,
            'stopping_reason': stopping_reason,
            'sample_hash': sample_hash,
            'sample_n': len(X),
            'init_sec': float(init_sec),
            'cluster_sec': float(cluster_sec),
            'total_sec': float(total_sec),
        })

        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
        el.save_csv(raw, RAW_PATH)

actual_pairs = set(
    zip(
        pd.to_numeric(raw['rep'], errors='coerce').astype(int),
        raw['variant_id'].astype(str),
    )
)
missing = sorted(expected_pairs - actual_pairs)
if missing:
    raise RuntimeError(f'Ablation checkpoint incomplete; missing {len(missing)} runs, e.g. {missing[:5]}')

if raw['rep'].nunique() != REPS:
    raise RuntimeError(f'Expected {REPS} repetitions, got {raw["rep"].nunique()}')

hash_counts = raw.groupby('rep')['sample_hash'].nunique()
if not (hash_counts == 1).all():
    raise RuntimeError('Matched-resample integrity FAIL: variants within a rep used different samples.')

counts = raw.groupby(['rep','variant_id']).size()
if not (counts == 1).all():
    raise RuntimeError('Duplicate (rep, variant) rows detected.')

print('Ablation completeness: PASS')
print('Matched-resample integrity: PASS')
print('Raw rows:', len(raw))
print(raw.head())


agg = {
    'n': ('rep', 'size'),
    'initial_cost_mean': ('initial_cost', 'mean'),
    'initial_cost_sd': ('initial_cost', 'std'),
    'final_cost_mean': ('final_cost', 'mean'),
    'final_cost_sd': ('final_cost', 'std'),
    'dbi_mean': ('dbi', 'mean'),
    'dbi_sd': ('dbi', 'std'),
    'si_mean': ('si', 'mean'),
    'si_sd': ('si', 'std'),
    'iterations_mean': ('iterations', 'mean'),
    'iterations_sd': ('iterations', 'std'),
    'accepted_corrections_mean': ('accepted_corrections', 'mean'),
    'early_stop_rate': ('early_stop', 'mean'),
    'init_sec_mean': ('init_sec', 'mean'),
    'init_sec_sd': ('init_sec', 'std'),
    'total_sec_mean': ('total_sec', 'mean'),
    'total_sec_sd': ('total_sec', 'std'),
}

summary = (
    raw.groupby(['variant_id', 'variant'], dropna=False)
       .agg(**agg)
       .reset_index()
)

order_map = {vid: i for i, vid in enumerate(VARIANT_ORDER)}
summary['order'] = summary['variant_id'].map(order_map)
summary = summary.sort_values('order').drop(columns='order').reset_index(drop=True)

el.save_csv(summary, SUMMARY_PATH)
print(summary)
print('Saved:', SUMMARY_PATH)


paper_table = summary[[
    'variant_id', 'variant',
    'initial_cost_mean', 'final_cost_mean',
    'dbi_mean', 'si_mean',
    'iterations_mean',
    'init_sec_mean', 'total_sec_mean',
    'accepted_corrections_mean',
]].copy()

paper_table = paper_table.rename(columns={
    'initial_cost_mean': 'Initial cost',
    'final_cost_mean': 'Final cost',
    'dbi_mean': 'DBI',
    'si_mean': 'SI',
    'iterations_mean': 'Iter.',
    'init_sec_mean': 'Init time (s)',
    'total_sec_mean': 'Total time (s)',
    'accepted_corrections_mean': 'Accepted corr.',
})

el.save_csv(paper_table, TABLE_PATH)
print(paper_table)
print('Saved:', TABLE_PATH)


METRIC_DIRECTIONS = {
    'initial_cost': 'lower',
    'final_cost': 'lower',
    'dbi': 'lower',
    'si': 'higher',
    'iterations': 'lower',
    'init_sec': 'lower',
    'total_sec': 'lower',
}

PLANNED_CONTRASTS = [
    {
        'contrast_id': 'C0',
        'contrast': 'Full CARE-KM vs Cao parent',
        'left': 'A4', 'right': 'A0',
        'component_question': 'net contribution beyond Cao parent',
    },
    {
        'contrast_id': 'C1',
        'contrast': 'Region effect under density screening',
        'left': 'A2', 'right': 'A1',
        'component_question': 'provisional-region restriction',
    },
    {
        'contrast_id': 'C2',
        'contrast': 'Region effect under separation screening',
        'left': 'A4', 'right': 'A3',
        'component_question': 'provisional-region restriction',
    },
    {
        'contrast_id': 'C3',
        'contrast': 'Separation effect without region restriction',
        'left': 'A3', 'right': 'A1',
        'component_question': 'separation-based candidate screening',
    },
    {
        'contrast_id': 'C4',
        'contrast': 'Separation effect with region restriction',
        'left': 'A4', 'right': 'A2',
        'component_question': 'separation-based candidate screening',
    },
]

contrast_rows = []

for metric, direction in METRIC_DIRECTIONS.items():
    metric_records = []

    for c in PLANNED_CONTRASTS:
        left = (
            raw.loc[raw['variant_id'].eq(c['left']), ['rep', metric]]
               .set_index('rep')[metric]
               .sort_index()
        )
        right = (
            raw.loc[raw['variant_id'].eq(c['right']), ['rep', metric]]
               .set_index('rep')[metric]
               .sort_index()
        )
        common = left.index.intersection(right.index)
        a = left.loc[common].to_numpy(dtype=float)
        b = right.loc[common].to_numpy(dtype=float)
        diff = a - b

        if direction == 'lower':
            wins = int(np.sum(a < b))
            losses = int(np.sum(a > b))
        else:
            wins = int(np.sum(a > b))
            losses = int(np.sum(a < b))
        ties = int(len(a) - wins - losses)

        if np.all(np.isclose(diff, 0.0, atol=1e-12, rtol=0.0)):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')

        metric_records.append({
            **c,
            'metric': metric,
            'direction': direction,
            'n_pairs': len(common),
            'left_mean': float(np.mean(a)),
            'right_mean': float(np.mean(b)),
            'mean_difference_left_minus_right': float(np.mean(diff)),
            'median_difference_left_minus_right': float(np.median(diff)),
            'left_wins': wins,
            'ties': ties,
            'left_losses': losses,
            'wilcoxon_statistic': float(stat),
            'wilcoxon_p_two_sided': float(p),
        })

    adjusted = el.holm_adjust([r['wilcoxon_p_two_sided'] for r in metric_records])
    for r, p_adj in zip(metric_records, adjusted):
        r['holm_p_within_metric'] = float(p_adj)
        contrast_rows.append(r)

contrasts = pd.DataFrame(contrast_rows)
el.save_csv(contrasts, CONTRAST_PATH)
print(contrasts)
print('Saved:', CONTRAST_PATH)


full_pairwise_rows = []
FULL_ID = 'A4'

for metric, direction in METRIC_DIRECTIONS.items():
    full = (
        raw.loc[raw['variant_id'].eq(FULL_ID), ['rep', metric]]
           .set_index('rep')[metric]
           .sort_index()
    )
    metric_rows = []

    for cmp_id in ['A0', 'A1', 'A2', 'A3']:
        comp = (
            raw.loc[raw['variant_id'].eq(cmp_id), ['rep', metric]]
               .set_index('rep')[metric]
               .sort_index()
        )
        common = full.index.intersection(comp.index)
        a = full.loc[common].to_numpy(dtype=float)
        b = comp.loc[common].to_numpy(dtype=float)
        diff = a - b

        if direction == 'lower':
            wins = int(np.sum(a < b))
            losses = int(np.sum(a > b))
        else:
            wins = int(np.sum(a > b))
            losses = int(np.sum(a < b))
        ties = int(len(a) - wins - losses)

        if np.all(np.isclose(diff, 0.0, atol=1e-12, rtol=0.0)):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')

        metric_rows.append({
            'metric': metric,
            'direction': direction,
            'full_variant': FULL_ID,
            'comparator_variant': cmp_id,
            'comparator_name': VARIANT_MAP[cmp_id]['variant'],
            'n_pairs': len(common),
            'full_mean': float(np.mean(a)),
            'comparator_mean': float(np.mean(b)),
            'mean_difference_full_minus_comparator': float(np.mean(diff)),
            'full_wins': wins,
            'ties': ties,
            'full_losses': losses,
            'wilcoxon_statistic': float(stat),
            'wilcoxon_p_two_sided': float(p),
        })

    adjusted = el.holm_adjust([r['wilcoxon_p_two_sided'] for r in metric_rows])
    for r, p_adj in zip(metric_rows, adjusted):
        r['holm_p_within_metric'] = float(p_adj)
        full_pairwise_rows.append(r)

full_pairwise = pd.DataFrame(full_pairwise_rows)
el.save_csv(full_pairwise, FULL_PAIRWISE_PATH)
print(full_pairwise)
print('Saved:', FULL_PAIRWISE_PATH)


expected_files = [
    RAW_PATH,
    SUMMARY_PATH,
    FULL_PAIRWISE_PATH,
    CONTRAST_PATH,
    TABLE_PATH,
    PROTOCOL_PATH,

]
missing = [str(p) for p in expected_files if not p.exists()]
if missing:
    raise RuntimeError(f'Missing expected ablation artifacts: {missing}')

if int(summary['n'].min()) != REPS or int(summary['n'].max()) != REPS:
    raise RuntimeError('Summary does not contain exactly 20 matched runs per variant.')

full_meta = el.frozen_care_metadata()
if int(full_meta['candidate_budget']) != Q:
    raise RuntimeError('Frozen CARE candidate budget changed unexpectedly.')
if int(full_meta['derived_refinement_budget']) != DERIVED_BUDGET:
    raise RuntimeError('Frozen derived refinement budget changed unexpectedly.')
if bool(full_meta.get('exchange_depth_is_independent_hyperparameter', True)):
    raise RuntimeError('Exchange depth is unexpectedly marked as an independent hyperparameter.')

print('TRUE COMPONENT ABLATION: PASS')
print('Artifacts:')
for p in expected_files:
    print(' -', p)
