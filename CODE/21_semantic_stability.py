# Mengukur kestabilan semantic direction pada matched resamples.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '07_STABILITY'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el
import care_km as ck

CARE_CONFIG = dict(el.FROZEN_CARE_CONFIG)
K = el.K

REPS = 30
FRACTION = 0.80
BASE_SEED = 6161600
BOOTSTRAP_SEED = 20261014
N_BOOT = 10000

CHECKPOINT_DIR = OUT_DIR / 'SEMANTIC_STABILITY_CHECKPOINTS'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print('PAPER_ROOT:', PAPER_ROOT)
print('K:', K)
print('CARE config:', CARE_CONFIG)
print(f'Protocol: {REPS} matched resamples x {FRACTION:.0%}')

df, X_all, FEATURES = el.load_lms_dataset(PAPER_ROOT)
df = df.copy()
df['user'] = df['user'].astype(str)
user_ids = df['user'].to_numpy()

if df['user'].duplicated().any():
    raise RuntimeError('Kolom user harus unik pada dataset CARE-KM.')

REF_FILE = PAPER_ROOT / 'OUTPUT' / '06_FSLSM_PROFILES' / '08_theory_guided_pseudolabeled_dataset.csv'
EVIDENCE_FILE = PAPER_ROOT / 'OUTPUT' / '06_FSLSM_PROFILES' / '08_semantic_evidence_matrix.csv'
ROBUST_FILE = PAPER_ROOT / 'OUTPUT' / '04_ROBUSTNESS' / '06_robustness_raw.csv'

for required in [REF_FILE, EVIDENCE_FILE]:
    if not required.is_file():
        raise FileNotFoundError(
            f'File wajib belum tersedia: {required.name}. '
            'Jalankan 08_fslsm_semantic_interpretation.py terlebih dahulu.'
        )

reference = pd.read_csv(REF_FILE, dtype={'user': str})
evidence = pd.read_csv(EVIDENCE_FILE)

required_evidence_cols = {
    'feature', 'dimension', 'low_supports', 'high_supports'
}
if not required_evidence_cols.issubset(evidence.columns):
    raise RuntimeError('08_semantic_evidence_matrix.csv tidak memiliki kolom yang dibutuhkan.')

missing_evidence = sorted(set(FEATURES) - set(evidence['feature'].astype(str)))
if missing_evidence:
    raise RuntimeError(f'Feature tanpa semantic evidence: {missing_evidence}')

if set(df['user']) != set(reference['user']):
    raise RuntimeError('User pada full-data reference tidak sama dengan dataset CARE-KM saat ini.')

reference = reference.set_index('user').loc[df['user']].reset_index()

robust06 = None
if ROBUST_FILE.is_file():
    robust06 = pd.read_csv(ROBUST_FILE)
    robust06 = robust06[robust06['method'].astype(str).eq('CARE-KM')].copy()
    robust06['rep'] = pd.to_numeric(robust06['rep'], errors='raise').astype(int)
    robust06 = robust06.sort_values('rep')
    if len(robust06) != REPS:
        raise RuntimeError('06_robustness_raw.csv tidak memiliki tepat 30 row CARE-KM.')
    print('06_lms_robustness_30x80.py integrity reference: FOUND')
else:
    print('06_lms_robustness_30x80.py integrity reference: not found; hash cross-check will be skipped.')

print('Rows:', len(df))
print('Features:', len(FEATURES))
print('Reference LS classes:', reference['ls_class'].nunique())

DIMENSIONS = [
    ('Active/Reflective', ('Active', 'Reflective')),
    ('Sensing/Intuitive', ('Sensing', 'Intuitive')),
    ('Visual/Verbal', ('Visual', 'Verbal')),
    ('Sequential/Global', ('Sequential', 'Global')),
]

POLES = {name: pair for name, pair in DIMENSIONS}

ABBR = {
    'Active': 'Act.',
    'Reflective': 'Ref.',
    'Sensing': 'Sens.',
    'Intuitive': 'Int.',
    'Visual': 'Vis.',
    'Verbal': 'Ver.',
    'Sequential': 'Seq.',
    'Global': 'Gl.',
}

DIM_KEY = {
    'Active/Reflective': 'active_reflective',
    'Sensing/Intuitive': 'sensing_intuitive',
    'Visual/Verbal': 'visual_verbal',
    'Sequential/Global': 'sequential_global',
}

ALLOWED_MAGNITUDES = {1, 3, 5, 7, 9, 11}

LS_LOOKUP = {}
for idx, combo in enumerate(
    itertools.product(*[pair for _, pair in DIMENSIONS]),
    start=1,
):
    LS_LOOKUP[tuple(combo)] = f'LS{idx:02d}'


def signed_vote(value, row):
    value = int(value)
    first, second = POLES[str(row.dimension)]

    if value == 2:
        return 0

    pole = str(row.high_supports) if value == 3 else str(row.low_supports)

    if pole == first:
        return 1
    if pole == second:
        return -1

    raise ValueError(f'Pole tidak dikenal: {pole}')


def nearest_integer(value):
    return int(np.floor(float(value) + 0.5))


def ils_equivalent_score(positive, negative, neutral):
    total = int(positive + negative + neutral)

    if total <= 0:
        return {
            'raw_score': np.nan,
            'magnitude': np.nan,
            'category': 'Unresolved',
        }

    positive_adjusted = float(positive) + 0.5 * float(neutral)
    positive_units = nearest_integer(11.0 * positive_adjusted / total)
    positive_units = min(11, max(0, positive_units))
    negative_units = 11 - positive_units

    raw_score = int(positive_units - negative_units)
    magnitude = abs(raw_score)

    if magnitude not in ALLOWED_MAGNITUDES:
        raise RuntimeError(f'Skor 11-equivalent tidak valid: {raw_score}')

    if magnitude in {1, 3}:
        category = 'Balanced'
    elif magnitude in {5, 7}:
        category = 'Moderate preference'
    else:
        category = 'Strong preference'

    return {
        'raw_score': raw_score,
        'magnitude': magnitude,
        'category': category,
    }


def interpret_partition(X, modes, labels, features, evidence_matrix):

    X = np.asarray(X, dtype=object)
    modes = np.asarray(modes, dtype=object)
    labels = np.asarray(labels, dtype=int)

    if modes.shape != (K, len(features)):
        raise RuntimeError(f'Mode shape tidak valid: {modes.shape}')

    fpos = {f: j for j, f in enumerate(features)}
    cluster_sizes = np.bincount(labels, minlength=K)
    rows = []

    for cid in range(K):
        selected_full = []
        selected_abbr = []
        scores = []
        magnitudes = []
        categories = []
        codes = []
        sources = []
        consistencies = []

        for dimension, _ in DIMENSIONS:
            rules = evidence_matrix[
                evidence_matrix['dimension'].astype(str).eq(dimension)
            ]
            first, second = POLES[dimension]
            mode_row = modes[cid]

            mode_votes = np.asarray([
                signed_vote(mode_row[fpos[str(r.feature)]], r)
                for r in rules.itertuples(index=False)
            ], dtype=int)

            mode_positive = int(np.sum(mode_votes > 0))
            mode_negative = int(np.sum(mode_votes < 0))
            mode_score = int(mode_positive - mode_negative)

            member_X = X[labels == cid]
            member_votes = []

            for r in rules.itertuples(index=False):
                col = fpos[str(r.feature)]
                member_votes.extend(
                    signed_vote(v, r) for v in member_X[:, col]
                )

            member_votes = np.asarray(member_votes, dtype=int)
            member_positive = int(np.sum(member_votes > 0))
            member_negative = int(np.sum(member_votes < 0))
            member_neutral = int(np.sum(member_votes == 0))
            member_score = int(member_positive - member_negative)

            if mode_score > 0:
                selected = first
                source = 'MODE'
            elif mode_score < 0:
                selected = second
                source = 'MODE'
            elif member_score > 0:
                selected = first
                source = 'MEMBER'
            elif member_score < 0:
                selected = second
                source = 'MEMBER'
            else:
                selected = None
                source = 'BALANCED'

            member_direction = (
                first if member_score > 0
                else second if member_score < 0
                else None
            )

            score11 = ils_equivalent_score(
                member_positive,
                member_negative,
                member_neutral,
            )

            if selected is None:
                selected_abbreviation = 'Balanced'
                signed_score = np.nan
                preference_code = 'Balanced'
                consistency = True
            else:
                selected_abbreviation = ABBR[selected]
                signed_score = (
                    score11['magnitude']
                    if selected == first
                    else -score11['magnitude']
                )
                preference_code = (
                    f"{selected_abbreviation}{score11['magnitude']}"
                )
                consistency = (
                    member_direction is None
                    or member_direction == selected
                )

            selected_full.append(selected)
            selected_abbr.append(selected_abbreviation)
            scores.append(signed_score)
            magnitudes.append(score11['magnitude'])
            categories.append(score11['category'])
            codes.append(preference_code)
            sources.append(source)
            consistencies.append(bool(consistency))

        unresolved = any(x is None for x in selected_full)

        if unresolved:
            ls_class = 'AMBIGUOUS'
            semantic_label = '-'.join(selected_abbr)
            preference_signature = '|'.join(codes)
            mean_strength = np.nan
        else:
            ls_class = LS_LOOKUP[tuple(selected_full)]
            semantic_label = '-'.join(ABBR[x] for x in selected_full)
            preference_signature = '|'.join(codes)
            mean_strength = float(np.mean(magnitudes))

        rows.append({
            'cluster_id': cid,
            'cluster_name': f'C{cid + 1:02d}',
            'cluster_size': int(cluster_sizes[cid]),
            'active_reflective': selected_abbr[0],
            'sensing_intuitive': selected_abbr[1],
            'visual_verbal': selected_abbr[2],
            'sequential_global': selected_abbr[3],
            'active_reflective_score': scores[0],
            'sensing_intuitive_score': scores[1],
            'visual_verbal_score': scores[2],
            'sequential_global_score': scores[3],
            'active_reflective_category': categories[0],
            'sensing_intuitive_category': categories[1],
            'visual_verbal_category': categories[2],
            'sequential_global_category': categories[3],
            'active_reflective_code': codes[0],
            'sensing_intuitive_code': codes[1],
            'visual_verbal_code': codes[2],
            'sequential_global_code': codes[3],
            'semantic_label': semantic_label,
            'ls_class': ls_class,
            'preference_signature': preference_signature,
            'mean_behavioral_preference_strength': mean_strength,
            'mapping_status': 'BALANCED' if unresolved else 'RESOLVED',
            'mode_dimensions': int(sum(x == 'MODE' for x in sources)),
            'member_dimensions': int(sum(x == 'MEMBER' for x in sources)),
            'balanced_dimensions': int(sum(x == 'BALANCED' for x in sources)),
            'mode_member_disagreements': int(sum(not x for x in consistencies)),
        })

    return pd.DataFrame(rows)


def bootstrap_mean_ci(values, seed, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    )
    boot_means = samples.mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)

direction_cols = [
    'active_reflective',
    'sensing_intuitive',
    'visual_verbal',
    'sequential_global',
]

score_cols = [
    'active_reflective_score',
    'sensing_intuitive_score',
    'visual_verbal_score',
    'sequential_global_score',
]

reference_cols = [
    'user',
    'ls_class',
    'semantic_label',
    'mapping_status',
    *direction_cols,
    *score_cols,
]

missing_ref = [c for c in reference_cols if c not in reference.columns]
if missing_ref:
    raise RuntimeError(f'Kolom reference tidak lengkap: {missing_ref}')

ref_small = reference[reference_cols].copy()
ref_small = ref_small.rename(
    columns={c: f'{c}_ref' for c in reference_cols if c != 'user'}
)

print('Reference mapping status:')
print(
    reference['mapping_status']
    .value_counts(dropna=False)
    .rename_axis('status')
    .reset_index(name='n')
)

def compare_members(rep, idx, sample_hash, result, mapping):
    labels = np.asarray(result['labels'], dtype=int)

    member = pd.DataFrame({
        'rep': rep,
        'sample_hash': sample_hash,
        'row_index': idx,
        'user': user_ids[idx],
        'cluster_id_resample': labels,
    })

    map_cols = [
        'cluster_id',
        'ls_class',
        'semantic_label',
        'mapping_status',
        *direction_cols,
        *score_cols,
    ]
    current = mapping[map_cols].copy()
    current = current.rename(
        columns={
            c: f'{c}_resample'
            for c in map_cols
            if c != 'cluster_id'
        }
    )

    member = member.merge(
        current,
        left_on='cluster_id_resample',
        right_on='cluster_id',
        how='left',
        validate='many_to_one',
    ).drop(columns='cluster_id')

    member = member.merge(
        ref_small,
        on='user',
        how='left',
        validate='one_to_one',
    )

    if member['ls_class_ref'].isna().any():
        raise RuntimeError(f'Reference join gagal pada rep {rep}.')

    member['base_ls_agreement'] = (
        member['ls_class_resample'].astype(str)
        == member['ls_class_ref'].astype(str)
    )

    member['semantic_label_agreement'] = (
        member['semantic_label_resample'].astype(str)
        == member['semantic_label_ref'].astype(str)
    )

    direction_agreement_cols = []
    score_exact_cols = []
    score_abs_diff_cols = []

    for direction_col, score_col in zip(direction_cols, score_cols):
        agree_col = f'{direction_col}_agreement'
        exact_col = f'{score_col}_exact'
        diff_col = f'{score_col}_abs_diff'

        member[agree_col] = (
            member[f'{direction_col}_resample'].astype(str)
            == member[f'{direction_col}_ref'].astype(str)
        )

        score_resample = pd.to_numeric(
            member[f'{score_col}_resample'],
            errors='coerce',
        )
        score_reference = pd.to_numeric(
            member[f'{score_col}_ref'],
            errors='coerce',
        )

        valid_scores = score_resample.notna() & score_reference.notna()
        member[exact_col] = False
        member.loc[valid_scores, exact_col] = np.isclose(
            score_resample[valid_scores],
            score_reference[valid_scores],
        )

        member[diff_col] = (
            score_resample - score_reference
        ).abs()

        direction_agreement_cols.append(agree_col)
        score_exact_cols.append(exact_col)
        score_abs_diff_cols.append(diff_col)

    member['all_four_direction_agreement'] = (
        member[direction_agreement_cols].all(axis=1)
    )

    member['mean_direction_agreement'] = (
        member[direction_agreement_cols].mean(axis=1)
    )

    member['mean_score_exact_agreement'] = (
        member[score_exact_cols].mean(axis=1)
    )

    member['mean_abs_score_change'] = (
        member[score_abs_diff_cols].mean(axis=1)
    )

    member['resample_unresolved'] = (
        member['mapping_status_resample'].astype(str) != 'RESOLVED'
    )

    return member


def summarize_rep(rep, sample_hash, row, mapping, member):
    summary = {
        'rep': rep,
        'sample_hash': sample_hash,
        'sample_n': int(len(member)),
        'final_cost': float(row['final_cost']),
        'dbi': float(row['dbi']),
        'si': float(row['si']),
        'iterations': int(row['iterations']),
        'base_ls_agreement': float(member['base_ls_agreement'].mean()),
        'semantic_label_agreement': float(
            member['semantic_label_agreement'].mean()
        ),
        'all_four_direction_agreement': float(
            member['all_four_direction_agreement'].mean()
        ),
        'mean_direction_agreement': float(
            member['mean_direction_agreement'].mean()
        ),
        'mean_score_exact_agreement': float(
            member['mean_score_exact_agreement'].mean()
        ),
        'mean_abs_score_change': float(
            member['mean_abs_score_change'].mean()
        ),
        'unresolved_learner_rate': float(
            member['resample_unresolved'].mean()
        ),
        'resolved_clusters': int(
            mapping['mapping_status'].eq('RESOLVED').sum()
        ),
        'unresolved_clusters': int(
            mapping['mapping_status'].ne('RESOLVED').sum()
        ),
        'distinct_ls_classes': int(
            mapping.loc[
                mapping['mapping_status'].eq('RESOLVED'),
                'ls_class',
            ].nunique()
        ),
        'mode_resolved_dimensions': int(mapping['mode_dimensions'].sum()),
        'member_resolved_dimensions': int(mapping['member_dimensions'].sum()),
        'balanced_dimensions': int(mapping['balanced_dimensions'].sum()),
        'mode_member_disagreements': int(
            mapping['mode_member_disagreements'].sum()
        ),
    }

    for direction_col, score_col in zip(direction_cols, score_cols):
        summary[f'{direction_col}_agreement'] = float(
            member[f'{direction_col}_agreement'].mean()
        )
        summary[f'{score_col}_exact_agreement'] = float(
            member[f'{score_col}_exact'].mean()
        )
        summary[f'{score_col}_mae'] = float(
            member[f'{score_col}_abs_diff'].mean()
        )

    return summary

for rep in range(REPS):
    summary_path = CHECKPOINT_DIR / f'rep_{rep:02d}_summary.json'
    mapping_path = CHECKPOINT_DIR / f'rep_{rep:02d}_cluster_mapping.csv'
    member_path = CHECKPOINT_DIR / f'rep_{rep:02d}_member_comparison.csv'

    if summary_path.exists() and mapping_path.exists() and member_path.exists():
        print(f'[{rep + 1:02d}/{REPS}] checkpoint exists')
        continue

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

    expected = None
    if robust06 is not None:
        expected_row = robust06.loc[robust06['rep'].eq(rep)]
        if len(expected_row) != 1:
            raise RuntimeError(
                f'CARE-KM robustness reference tidak unik pada rep {rep}.'
            )
        expected = expected_row.iloc[0]

        if str(expected['sample_hash']) != str(sample_hash):
            raise RuntimeError(
                f'Sample hash berbeda pada rep {rep}: '
                f"{sample_hash} != {expected['sample_hash']}"
            )

    row, _, result = ck.run_method(
        'CARE-KM',
        X,
        None,
        K,
        seed=BASE_SEED + rep,
        care_config=CARE_CONFIG,
        si_sample_size=el.SI_SAMPLE_SIZE,
        si_seed=BASE_SEED + rep,
        max_iter=el.MAX_ITER,
    )

    if expected is not None:
        checks = {
            'final_cost': (
                float(row['final_cost']),
                float(expected['final_cost']),
            ),
            'dbi': (
                float(row['dbi']),
                float(expected['dbi']),
            ),
            'si': (
                float(row['si']),
                float(expected['si']),
            ),
            'iterations': (
                float(row['iterations']),
                float(expected['iterations']),
            ),
        }

        for metric, (observed, target) in checks.items():


            if metric in {'final_cost', 'iterations'}:
                matched = np.isclose(
                    observed,
                    target,
                    rtol=0.0,
                    atol=1e-12,
                )
            else:
                matched = np.isclose(
                    observed,
                    target,
                    rtol=0.0,
                    atol=1e-6,
                )

            if not matched:
                raise RuntimeError(
                    f'CARE-KM result berbeda dari 06_lms_robustness_30x80.py pada '
                    f'rep={rep}, metric={metric}: {observed} != {target}'
                )

    mapping = interpret_partition(
        X,
        result['modes'],
        result['labels'],
        FEATURES,
        evidence,
    )
    mapping.insert(0, 'rep', rep)
    mapping.insert(1, 'sample_hash', sample_hash)

    member = compare_members(
        rep,
        idx,
        sample_hash,
        result,
        mapping,
    )

    summary = summarize_rep(
        rep,
        sample_hash,
        row,
        mapping,
        member,
    )

    keep_member_cols = [
        'rep', 'sample_hash', 'row_index', 'user',
        'cluster_id_resample',
        'ls_class_ref', 'ls_class_resample',
        'semantic_label_ref', 'semantic_label_resample',
        'mapping_status_ref', 'mapping_status_resample',
        'base_ls_agreement',
        'all_four_direction_agreement',
        'mean_direction_agreement',
        'mean_score_exact_agreement',
        'mean_abs_score_change',
        'resample_unresolved',
    ]

    for direction_col, score_col in zip(direction_cols, score_cols):
        keep_member_cols.extend([
            f'{direction_col}_ref',
            f'{direction_col}_resample',
            f'{direction_col}_agreement',
            f'{score_col}_ref',
            f'{score_col}_resample',
            f'{score_col}_exact',
            f'{score_col}_abs_diff',
        ])

    el.save_csv(
        mapping,
        mapping_path,
        index=False,
    )
    el.save_csv(
        member[keep_member_cols],
        member_path,
        index=False,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )

    print(
        f"[{rep + 1:02d}/{REPS}] "
        f"hash={sample_hash} | "
        f"Base-LS={summary['base_ls_agreement']:.4f} | "
        f"4D={summary['all_four_direction_agreement']:.4f} | "
        f"unresolved={summary['unresolved_learner_rate']:.4f}"
    )

print('30-resample semantic stability run complete.')

summary_rows = []
mapping_frames = []
member_frames = []

for rep in range(REPS):
    summary_path = CHECKPOINT_DIR / f'rep_{rep:02d}_summary.json'
    mapping_path = CHECKPOINT_DIR / f'rep_{rep:02d}_cluster_mapping.csv'
    member_path = CHECKPOINT_DIR / f'rep_{rep:02d}_member_comparison.csv'

    if not (
        summary_path.exists()
        and mapping_path.exists()
        and member_path.exists()
    ):
        raise RuntimeError(f'Checkpoint rep {rep} belum lengkap.')

    summary_rows.append(
        json.loads(summary_path.read_text(encoding='utf-8'))
    )
    mapping_frames.append(
        pd.read_csv(mapping_path, dtype={'sample_hash': str})
    )
    member_frames.append(
        pd.read_csv(
            member_path,
            dtype={'user': str, 'sample_hash': str},
        )
    )

resample_summary = (
    pd.DataFrame(summary_rows)
    .sort_values('rep')
    .reset_index(drop=True)
)
cluster_mapping = pd.concat(mapping_frames, ignore_index=True)
member_comparison = pd.concat(member_frames, ignore_index=True)

if len(resample_summary) != REPS:
    raise RuntimeError('Jumlah resample summary bukan 30.')

if resample_summary['sample_hash'].nunique() != REPS:
    raise RuntimeError('Sample hash seharusnya unik untuk setiap repetition.')

if len(cluster_mapping) != REPS * K:
    raise RuntimeError('Jumlah cluster mapping tidak sama dengan 30 x K.')

expected_member_rows = int(len(X_all) * FRACTION) * REPS
if len(member_comparison) != expected_member_rows:
    raise RuntimeError(
        f'Jumlah member comparison tidak sesuai: '
        f'{len(member_comparison)} != {expected_member_rows}'
    )

el.save_csv(
    resample_summary,
    OUT_DIR / 'semantic_stability_resample_summary.csv',
    index=False,
)
el.save_csv(
    cluster_mapping,
    OUT_DIR / 'semantic_stability_cluster_mapping.csv',
    index=False,
)
el.save_csv(
    member_comparison,
    OUT_DIR / 'semantic_stability_member_comparison.csv',
    index=False,
)

print(resample_summary)

metric_cols = [
    'base_ls_agreement',
    'semantic_label_agreement',
    'all_four_direction_agreement',
    'mean_direction_agreement',
    'mean_score_exact_agreement',
    'mean_abs_score_change',
    'unresolved_learner_rate',
    'active_reflective_agreement',
    'sensing_intuitive_agreement',
    'visual_verbal_agreement',
    'sequential_global_agreement',
    'active_reflective_score_exact_agreement',
    'sensing_intuitive_score_exact_agreement',
    'visual_verbal_score_exact_agreement',
    'sequential_global_score_exact_agreement',
    'active_reflective_score_mae',
    'sensing_intuitive_score_mae',
    'visual_verbal_score_mae',
    'sequential_global_score_mae',
]

metric_rows = []

for i, metric in enumerate(metric_cols):
    values = pd.to_numeric(
        resample_summary[metric],
        errors='coerce',
    ).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    ci_low, ci_high = bootstrap_mean_ci(
        finite,
        seed=BOOTSTRAP_SEED + i,
    )

    metric_rows.append({
        'metric': metric,
        'n_reps': int(len(finite)),
        'mean': float(np.mean(finite)),
        'sd': float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        'median': float(np.median(finite)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'bootstrap_95ci_low': ci_low,
        'bootstrap_95ci_high': ci_high,
    })

metric_summary = pd.DataFrame(metric_rows)

el.save_csv(
    metric_summary,
    OUT_DIR / 'semantic_stability_metric_summary.csv',
    index=False,
)

print(metric_summary)

dimension_rows = []

for dimension, _ in DIMENSIONS:
    key = DIM_KEY[dimension]

    agreement = pd.to_numeric(
        resample_summary[f'{key}_agreement'],
        errors='coerce',
    ).to_numpy(dtype=float)

    score_exact = pd.to_numeric(
        resample_summary[f'{key}_score_exact_agreement'],
        errors='coerce',
    ).to_numpy(dtype=float)

    score_mae = pd.to_numeric(
        resample_summary[f'{key}_score_mae'],
        errors='coerce',
    ).to_numpy(dtype=float)

    agreement_ci = bootstrap_mean_ci(
        agreement,
        seed=BOOTSTRAP_SEED + 100 + len(dimension_rows),
    )
    exact_ci = bootstrap_mean_ci(
        score_exact,
        seed=BOOTSTRAP_SEED + 200 + len(dimension_rows),
    )
    mae_ci = bootstrap_mean_ci(
        score_mae,
        seed=BOOTSTRAP_SEED + 300 + len(dimension_rows),
    )

    dimension_rows.append({
        'dimension': dimension,
        'direction_agreement_mean': float(np.mean(agreement)),
        'direction_agreement_sd': float(np.std(agreement, ddof=1)),
        'direction_agreement_95ci_low': agreement_ci[0],
        'direction_agreement_95ci_high': agreement_ci[1],
        'score_exact_agreement_mean': float(np.mean(score_exact)),
        'score_exact_agreement_sd': float(np.std(score_exact, ddof=1)),
        'score_exact_agreement_95ci_low': exact_ci[0],
        'score_exact_agreement_95ci_high': exact_ci[1],
        'score_mae_mean': float(np.mean(score_mae)),
        'score_mae_sd': float(np.std(score_mae, ddof=1)),
        'score_mae_95ci_low': mae_ci[0],
        'score_mae_95ci_high': mae_ci[1],
    })

dimension_summary = pd.DataFrame(dimension_rows)

el.save_csv(
    dimension_summary,
    OUT_DIR / 'semantic_stability_dimension_summary.csv',
    index=False,
)

print(dimension_summary)

ls_class_summary = (
    member_comparison
    .groupby('ls_class_ref', dropna=False)
    .agg(
        learner_resample_observations=('user', 'size'),
        unique_learners=('user', 'nunique'),
        base_ls_agreement=('base_ls_agreement', 'mean'),
        all_four_direction_agreement=(
            'all_four_direction_agreement',
            'mean',
        ),
        mean_direction_agreement=('mean_direction_agreement', 'mean'),
        mean_score_exact_agreement=(
            'mean_score_exact_agreement',
            'mean',
        ),
        mean_abs_score_change=('mean_abs_score_change', 'mean'),
        unresolved_rate=('resample_unresolved', 'mean'),
    )
    .reset_index()
    .sort_values(
        ['base_ls_agreement', 'ls_class_ref'],
        ascending=[False, True],
    )
)

el.save_csv(
    ls_class_summary,
    OUT_DIR / 'semantic_stability_ls_class_summary.csv',
    index=False,
)

transition_counts = pd.crosstab(
    member_comparison['ls_class_ref'],
    member_comparison['ls_class_resample'],
    dropna=False,
)

transition_prop = transition_counts.div(
    transition_counts.sum(axis=1),
    axis=0,
)

transition_long = (
    transition_counts
    .stack(future_stack=True)
    .rename('n')
    .reset_index()
    .merge(
        transition_prop
        .stack(future_stack=True)
        .rename('row_proportion')
        .reset_index(),
        on=['ls_class_ref', 'ls_class_resample'],
        how='left',
    )
)

el.save_csv(
    transition_long,
    OUT_DIR / 'semantic_stability_ls_transition.csv',
    index=False,
)

print(ls_class_summary)

integrity_rows = []

for rep in range(REPS):
    observed_hash = str(
        resample_summary.loc[
            resample_summary['rep'].eq(rep),
            'sample_hash',
        ].iloc[0]
    )

    expected_hash = None
    hash_match = None

    if robust06 is not None:
        expected_hash = str(
            robust06.loc[
                robust06['rep'].eq(rep),
                'sample_hash',
            ].iloc[0]
        )
        hash_match = observed_hash == expected_hash

    integrity_rows.append({
        'rep': rep,
        'observed_sample_hash': observed_hash,
        'notebook06_sample_hash': expected_hash,
        'hash_match': hash_match,
    })

sample_integrity = pd.DataFrame(integrity_rows)

if robust06 is not None and not sample_integrity['hash_match'].all():
    raise RuntimeError('Ada sample hash yang tidak sama dengan 06_lms_robustness_30x80.py.')

el.save_csv(
    sample_integrity,
    OUT_DIR / 'semantic_stability_sample_integrity.csv',
    index=False,
)


def metric_mean(name):
    return float(
        metric_summary.loc[
            metric_summary['metric'].eq(name),
            'mean',
        ].iloc[0]
    )


validation = {
    'analysis': 'FSLSM semantic-label stability across matched resamples',
    'reference': 'full-data CARE-KM semantic interpretation from 08_fslsm_semantic_interpretation.py',
    'resampling_protocol': '30 matched 80% learner subsamples',
    'repetitions': REPS,
    'fraction': FRACTION,
    'base_seed': BASE_SEED,
    'same_sampling_protocol_as_notebook_06': True,
    'sample_hash_verified_against_notebook_06': bool(
        robust06 is not None
        and sample_integrity['hash_match'].all()
    ),
    'semantic_feedback_to_clustering': False,
    'hungarian_assignment': False,
    'one_to_one_constraint': False,
    'questionnaire_ground_truth_claim': False,
    'base_ls_agreement_mean': metric_mean(
        'base_ls_agreement'
    ),
    'all_four_direction_agreement_mean': metric_mean(
        'all_four_direction_agreement'
    ),
    'mean_direction_agreement': metric_mean(
        'mean_direction_agreement'
    ),
    'mean_score_exact_agreement': metric_mean(
        'mean_score_exact_agreement'
    ),
    'mean_abs_score_change': metric_mean(
        'mean_abs_score_change'
    ),
    'unresolved_learner_rate_mean': metric_mean(
        'unresolved_learner_rate'
    ),
    'output_files': [
        'semantic_stability_resample_summary.csv',
        'semantic_stability_metric_summary.csv',
        'semantic_stability_dimension_summary.csv',
        'semantic_stability_ls_class_summary.csv',
        'semantic_stability_ls_transition.csv',
        'semantic_stability_cluster_mapping.csv',
        'semantic_stability_member_comparison.csv',
        'semantic_stability_sample_integrity.csv',
    ],
}

(OUT_DIR / 'semantic_stability_validation.json').write_text(
    json.dumps(validation, indent=2),
    encoding='utf-8',
)

print(json.dumps(validation, indent=2))
