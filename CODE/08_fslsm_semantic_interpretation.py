# Membentuk profil FSLSM setelah clustering selesai.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import json
import os
import sys
import hashlib
import itertools

import numpy as np
import pandas as pd


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
DATA_DIR = PAPER_ROOT / 'DATA'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '06_FSLSM_PROFILES'
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
user_ids = df['user'].astype(str).to_numpy()


def save_csv6(frame, path, index=False):
    el.save_csv(frame, path, index=index)


evidence_rows = []


def add_evidence(feature, dimension, low_supports, high_supports,
                 semantic_basis, evidence_type, source):
    evidence_rows.append({
        'feature': feature,
        'dimension': dimension,
        'low_supports': low_supports,
        'moderate_supports': 'Neutral',
        'high_supports': high_supports,
        'semantic_basis': semantic_basis,
        'evidence_type': evidence_type,
        'source': source,
    })


add_evidence('content_visit', 'Active/Reflective', 'Active', 'Reflective',
             'trying something out / think about material', 'direct',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('content_stay', 'Active/Reflective', 'Active', 'Reflective',
             'trying something out / think about material', 'direct',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('selfass_visit_proxy', 'Active/Reflective', 'Reflective', 'Active',
             'think about material / trying something out', 'proxy',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('selfass_stay_proxy', 'Active/Reflective', 'Active', 'Reflective',
             'trying something out / think about material', 'proxy',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('exercise_visit_proxy', 'Active/Reflective', 'Reflective', 'Active',
             'think about material / trying something out', 'proxy',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('exercise_stay_proxy', 'Active/Reflective', 'Reflective', 'Active',
             'think about material / trying something out', 'proxy',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('forum_visit', 'Active/Reflective', 'Active', 'Reflective',
             'social oriented / impersonal oriented', 'direct',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('forum_post', 'Active/Reflective', 'Reflective', 'Active',
             'impersonal oriented / social oriented', 'direct',
             'Graf 2007 Table 5.2; Table 5.10')
add_evidence('quiz_stay_results_proxy', 'Active/Reflective', 'Active', 'Reflective',
             'less review / more reflection on results', 'proxy',
             'Graf 2007 Table 5.2; Table 5.10')


add_evidence('content_visit', 'Sensing/Intuitive', 'Sensing', 'Intuitive',
             'concrete material / abstract material', 'direct',
             'Graf 2007 Table 5.2; Table 5.11')
add_evidence('content_stay', 'Sensing/Intuitive', 'Sensing', 'Intuitive',
             'concrete material / abstract material', 'direct',
             'Graf 2007 Table 5.2; Table 5.11')
add_evidence('selfass_visit_proxy', 'Sensing/Intuitive', 'Intuitive', 'Sensing',
             'new ways / existing ways', 'proxy',
             'Graf 2007 Table 5.2; Table 5.11')
add_evidence('selfass_stay_proxy', 'Sensing/Intuitive', 'Intuitive', 'Sensing',
             'not careful / careful with details', 'proxy',
             'Graf 2007 Table 5.2; Table 5.11')
add_evidence('exercise_visit_proxy', 'Sensing/Intuitive', 'Intuitive', 'Sensing',
             'new ways / existing ways', 'proxy',
             'Graf 2007 Table 5.2; Table 5.11')
add_evidence('quiz_stay_results_proxy', 'Sensing/Intuitive', 'Intuitive', 'Sensing',
             'not careful / careful with details', 'proxy',
             'Graf 2007 Table 5.2; Table 5.11')


add_evidence('content_visit', 'Visual/Verbal', 'Visual', 'Verbal',
             'pictures / written words', 'direct',
             'Graf 2007 Table 5.2; Table 5.12')
add_evidence('forum_visit', 'Visual/Verbal', 'Visual', 'Verbal',
             'pictures / written words', 'direct',
             'Graf 2007 Table 5.2; Table 5.12')
add_evidence('forum_stay', 'Visual/Verbal', 'Visual', 'Verbal',
             'pictures / written words', 'direct',
             'Graf 2007 Table 5.2; Table 5.12')
add_evidence('forum_post', 'Visual/Verbal', 'Visual', 'Verbal',
             'pictures / written words', 'direct',
             'Graf 2007 Table 5.2; Table 5.12')


add_evidence('navigation_overview_visit', 'Sequential/Global', 'Sequential', 'Global',
             'sequential progress / overall picture & non-sequential progress', 'direct',
             'Graf 2007 Table 5.2; Table 5.13')
add_evidence('navigation_overview_stay', 'Sequential/Global', 'Sequential', 'Global',
             'detail/sequential / overall picture', 'direct',
             'Graf 2007 Table 5.2; Table 5.13')
add_evidence('return_to_overview_rate', 'Sequential/Global', 'Sequential', 'Global',
             'sequential progress / non-sequential overview-return behavior', 'adapted',
             'Adapted from Graf 2007 overview/navigation semantics; Table 5.13')

evidence = pd.DataFrame(evidence_rows)
missing_evidence = sorted(set(FEATURES) - set(evidence['feature']))
if missing_evidence:
    raise RuntimeError(f'Feature tanpa semantic evidence: {missing_evidence}')

save_csv6(evidence, OUT_DIR / '08_semantic_evidence_matrix.csv')
evidence_bytes = evidence.to_csv(index=False).encode('utf-8')
EVIDENCE_SHA256 = hashlib.sha256(evidence_bytes).hexdigest()

print(evidence)
print('Semantic evidence SHA-256:', EVIDENCE_SHA256)


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
DIM_ABBR = {
    'Active/Reflective': 'A/R',
    'Sensing/Intuitive': 'Sens./Int.',
    'Visual/Verbal': 'Vis./Ver.',
    'Sequential/Global': 'Seq./Gl.',
}
ALLOWED_MAGNITUDES = {1, 3, 5, 7, 9, 11}

LS_LOOKUP = {}
for idx, combo in enumerate(itertools.product(*[pair for _, pair in DIMENSIONS]), start=1):
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
            'positive_adjusted': np.nan,
            'negative_adjusted': np.nan,
            'positive_units': np.nan,
            'negative_units': np.nan,
            'raw_score': np.nan,
            'magnitude': np.nan,
            'category': 'Unresolved',
        }

    positive_adjusted = float(positive) + 0.5 * float(neutral)
    negative_adjusted = float(negative) + 0.5 * float(neutral)

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
        'positive_adjusted': positive_adjusted,
        'negative_adjusted': negative_adjusted,
        'positive_units': int(positive_units),
        'negative_units': int(negative_units),
        'raw_score': raw_score,
        'magnitude': magnitude,
        'category': category,
    }


def mode_signature(mode):
    return '|'.join(str(v) for v in mode)

mode_file = PAPER_ROOT / 'OUTPUT' / '01_FULL_LMS' / '04_carekm_reference_final_modes.csv'
label_file = PAPER_ROOT / 'OUTPUT' / '01_FULL_LMS' / '04_carekm_reference_labels.csv'

if mode_file.exists() and label_file.exists():
    mode_df = pd.read_csv(mode_file, dtype=str)
    label_df = pd.read_csv(label_file, dtype={'user': str})
    modes = mode_df[FEATURES].astype(str).to_numpy(dtype=object)
    label_map = label_df.set_index('user')['cluster_id'].astype(int)
    labels = df['user'].astype(str).map(label_map).to_numpy(dtype=int)
    source = '04_lms_full_dataset_experiment.py'
else:
    init_modes = ck.care_init(X, K, **CARE_CONFIG)
    result = ck.kmodes_fit(X, init_modes, max_iter=el.MAX_ITER)
    modes = result['modes']
    labels = result['labels'].astype(int)
    source = '08 standalone CARE-KM rerun'

if modes.shape != (K, len(FEATURES)) or len(labels) != len(X):
    raise RuntimeError('Reference clustering output tidak valid.')

cluster_sizes = np.bincount(labels, minlength=K)
fpos = {f: j for j, f in enumerate(FEATURES)}

print('Clustering source:', source)

def dimension_decision(cluster_id, dimension):
    rules = evidence[evidence['dimension'].eq(dimension)]
    mode_row = modes[int(cluster_id)]
    first, second = POLES[dimension]

    mode_votes = np.asarray([
        signed_vote(mode_row[fpos[str(r.feature)]], r)
        for r in rules.itertuples(index=False)
    ], dtype=int)

    mode_positive = int(np.sum(mode_votes > 0))
    mode_negative = int(np.sum(mode_votes < 0))
    mode_neutral = int(np.sum(mode_votes == 0))
    mode_score = int(mode_positive - mode_negative)

    member_X = X[labels == int(cluster_id)]
    member_votes = []
    for r in rules.itertuples(index=False):
        col = fpos[str(r.feature)]
        member_votes.extend(signed_vote(v, r) for v in member_X[:, col])

    member_votes = np.asarray(member_votes, dtype=int)
    member_positive = int(np.sum(member_votes > 0))
    member_negative = int(np.sum(member_votes < 0))
    member_neutral = int(np.sum(member_votes == 0))
    member_score = int(member_positive - member_negative)

    if mode_score > 0:
        selected = first
        decision_source = 'MODE'
    elif mode_score < 0:
        selected = second
        decision_source = 'MODE'
    elif member_score > 0:
        selected = first
        decision_source = 'MEMBER'
    elif member_score < 0:
        selected = second
        decision_source = 'MEMBER'
    else:
        selected = None
        decision_source = 'BALANCED'

    member_direction = (
        first if member_score > 0
        else second if member_score < 0
        else None
    )
    score11 = ils_equivalent_score(
        member_positive, member_negative, member_neutral
    )

    if selected is None:
        selected_abbr = 'Balanced'
        signed_score = np.nan
        preference_code = 'Balanced'
        consistency = True
    else:
        selected_abbr = ABBR[selected]
        signed_score = (
            score11['magnitude'] if selected == first
            else -score11['magnitude']
        )
        preference_code = f"{selected_abbr}{score11['magnitude']}"
        consistency = member_direction is None or member_direction == selected

    return {
        'selected_pole': selected_abbr,
        'decision_source': decision_source,
        'mapped_features': int(len(rules)),
        'mode_positive': mode_positive,
        'mode_negative': mode_negative,
        'mode_neutral': mode_neutral,
        'mode_score': mode_score,
        'member_positive': member_positive,
        'member_negative': member_negative,
        'member_neutral': member_neutral,
        'member_score': member_score,
        'positive_adjusted_evidence': score11['positive_adjusted'],
        'negative_adjusted_evidence': score11['negative_adjusted'],
        'positive_11eq_units': score11['positive_units'],
        'negative_11eq_units': score11['negative_units'],
        'member_11eq_raw_score': score11['raw_score'],
        'preference_11eq_score': signed_score,
        'preference_11eq_magnitude': score11['magnitude'],
        'preference_category': score11['category'],
        'preference_code': preference_code,
        'mode_member_consistency': bool(consistency),
        '_selected_full': selected,
    }


dimension_rows = []
mapping_rows = []

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
        d = dimension_decision(cid, dimension)
        selected_full.append(d.pop('_selected_full'))

        dimension_rows.append({
            'cluster_id': cid,
            'cluster_name': f'C{cid+1:02d}',
            'cluster_size': int(cluster_sizes[cid]),
            'dimension': dimension,
            **d,
        })
        selected_abbr.append(d['selected_pole'])
        scores.append(d['preference_11eq_score'])
        magnitudes.append(d['preference_11eq_magnitude'])
        categories.append(d['preference_category'])
        codes.append(d['preference_code'])
        sources.append(d['decision_source'])
        consistencies.append(bool(d['mode_member_consistency']))

    unresolved = any(x is None for x in selected_full)

    if unresolved:
        ls_class = 'AMBIGUOUS'
        semantic_label = '-'.join(selected_abbr)
        preference_signature = '|'.join(codes)
        profile_variant = 'AMBIGUOUS'
        mean_strength = np.nan
        boundary_dimension = ''
        nearest_alternative_ls = ''
    else:
        ls_class = LS_LOOKUP[tuple(selected_full)]
        semantic_label = '-'.join(ABBR[x] for x in selected_full)
        preference_signature = '|'.join(codes)
        profile_variant = f"{ls_class} [{', '.join(codes)}]"
        mean_strength = float(np.mean(magnitudes))

        min_strength = min(magnitudes)
        boundary_idx = [
            i for i, value in enumerate(magnitudes)
            if value == min_strength
        ]
        boundary_dimension = '|'.join(
            DIM_ABBR[DIMENSIONS[i][0]] for i in boundary_idx
        )

        alternatives = []
        if min_strength in {1, 3}:
            for i in boundary_idx:
                alt = list(selected_full)
                first, second = DIMENSIONS[i][1]
                alt[i] = second if alt[i] == first else first
                alternatives.append(LS_LOOKUP[tuple(alt)])
        nearest_alternative_ls = '|'.join(dict.fromkeys(alternatives))

    mapping_rows.append({
        'cluster_id': cid,
        'cluster_name': f'C{cid+1:02d}',
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
        'behavioral_profile_variant': profile_variant,
        'mean_behavioral_preference_strength': mean_strength,
        'boundary_dimension': boundary_dimension,
        'nearest_alternative_ls': nearest_alternative_ls,
        'mapping_status': 'BALANCED' if unresolved else 'RESOLVED',
        'mode_dimensions': int(sum(s == 'MODE' for s in sources)),
        'member_dimensions': int(sum(s == 'MEMBER' for s in sources)),
        'balanced_dimensions': int(sum(s == 'BALANCED' for s in sources)),
        'mode_member_disagreements': int(sum(not x for x in consistencies)),
    })

cluster_dimension_evidence = pd.DataFrame(dimension_rows)
cluster_to_fslsm = pd.DataFrame(mapping_rows)

modes_df = pd.DataFrame(modes, columns=FEATURES)
modes_df.insert(0, 'cluster_name', [f'C{i+1:02d}' for i in range(K)])
modes_df.insert(0, 'cluster_id', np.arange(K))
modes_df['cluster_size'] = cluster_sizes
modes_df['mode_signature'] = [mode_signature(x) for x in modes]

mapping_by_cluster = cluster_to_fslsm.set_index('cluster_id')
final_df = df.copy()
final_df['final_cluster_id'] = labels
final_df['final_cluster_name'] = [f'C{x+1:02d}' for x in labels]

label_columns = [
    'ls_class', 'semantic_label', 'preference_signature',
    'behavioral_profile_variant',
    'active_reflective', 'sensing_intuitive',
    'visual_verbal', 'sequential_global',
    'active_reflective_score', 'sensing_intuitive_score',
    'visual_verbal_score', 'sequential_global_score',
    'active_reflective_category', 'sensing_intuitive_category',
    'visual_verbal_category', 'sequential_global_category',
    'active_reflective_code', 'sensing_intuitive_code',
    'visual_verbal_code', 'sequential_global_code',
    'mean_behavioral_preference_strength',
    'boundary_dimension', 'nearest_alternative_ls',
    'mapping_status',
]
for col in label_columns:
    final_df[col] = [
        mapping_by_cluster.loc[int(x), col] for x in labels
    ]

balanced_learners = final_df[
    final_df['mapping_status'].eq('BALANCED')
].copy()

distribution = (
    final_df.groupby(
        ['ls_class', 'semantic_label', 'preference_signature',
         'behavioral_profile_variant'],
        dropna=False,
    )
    .size()
    .reset_index(name='learners')
    .sort_values(['learners', 'ls_class'], ascending=[False, True])
)

save_csv6(modes_df, OUT_DIR / '08_final_cluster_modes.csv')
save_csv6(
    cluster_dimension_evidence,
    OUT_DIR / '08_cluster_dimension_evidence.csv'
)
save_csv6(cluster_to_fslsm, OUT_DIR / '08_cluster_to_fslsm_mapping.csv')
save_csv6(
    final_df,
    OUT_DIR / '08_theory_guided_pseudolabeled_dataset.csv'
)
save_csv6(distribution, OUT_DIR / '08_pseudolabel_distribution.csv')
save_csv6(distribution, OUT_DIR / '08_profile_variant_distribution.csv')
save_csv6(balanced_learners, OUT_DIR / '08_balanced_learners.csv')

print(cluster_to_fslsm)
print(distribution)

resolved = cluster_to_fslsm[
    cluster_to_fslsm['mapping_status'].eq('RESOLVED')
].copy()
variant_counts = resolved['behavioral_profile_variant'].value_counts()

score_cols = [
    'active_reflective_score',
    'sensing_intuitive_score',
    'visual_verbal_score',
    'sequential_global_score',
]
observed_scores = (
    resolved[score_cols]
    .apply(pd.to_numeric, errors='coerce')
    .abs()
    .stack()
    .dropna()
    .astype(int)
)
if not set(observed_scores.unique()).issubset(ALLOWED_MAGNITUDES):
    raise RuntimeError('Ada skor preferensi di luar skala 11-equivalent.')

strong_conflicts = cluster_dimension_evidence[
    (~cluster_dimension_evidence['mode_member_consistency'])
    & (cluster_dimension_evidence['preference_11eq_magnitude'] >= 5)
]

validation = {
    'clustering_source': source,
    'rows': int(len(df)),
    'K': K,
    'semantic_method': 'mode_first_ils_inspired_11_equivalent',
    'semantic_feedback_to_clustering': False,
    'label_type': 'Theory-Guided Pseudo-Label',
    'psychometric_ground_truth_claim': False,
    'behavioral_preference_strength': True,
    'strength_method': 'ils_inspired_11_equivalent_behavioral_preference',
    'equivalent_units_per_dimension': 11,
    'allowed_score_magnitudes': sorted(ALLOWED_MAGNITUDES),
    'preference_categories': {
        'Balanced': [1, 3],
        'Moderate preference': [5, 7],
        'Strong preference': [9, 11],
    },
    'member_evidence_used_for_strength': True,
    'neutral_evidence_split_symmetrically': True,
    'strength_is_ils_questionnaire_score': False,
    'mean_strength_is_ils_questionnaire_score': False,
    'member_fallback_only_when_mode_balanced': True,
    'nearest_prototype_for_labeling': False,
    'hungarian_assignment': False,
    'one_to_one_constraint': False,
    'bootstrap_for_labeling': False,
    'mode_resolved_dimensions': int(
        cluster_dimension_evidence['decision_source'].eq('MODE').sum()
    ),
    'member_resolved_dimensions': int(
        cluster_dimension_evidence['decision_source'].eq('MEMBER').sum()
    ),
    'balanced_dimensions': int(
        cluster_dimension_evidence['decision_source'].eq('BALANCED').sum()
    ),
    'mode_member_disagreements': int(
        (~cluster_dimension_evidence['mode_member_consistency']).sum()
    ),
    'strong_mode_member_conflicts': int(len(strong_conflicts)),
    'balanced_clusters': int(
        cluster_to_fslsm['mapping_status'].eq('BALANCED').sum()
    ),
    'balanced_learners': int(len(balanced_learners)),
    'distinct_directional_profiles': int(
        resolved['semantic_label'].nunique()
    ),
    'distinct_behavioral_profile_variants': int(
        resolved['behavioral_profile_variant'].nunique()
    ),
    'duplicate_profile_variant_groups': int((variant_counts > 1).sum()),
    'profile_variant_one_to_one_constraint': False,
    'semantic_evidence_sha256': EVIDENCE_SHA256,
}

(OUT_DIR / '08_semantic_validation.json').write_text(
    json.dumps(validation, indent=2),
    encoding='utf-8',
)

print(json.dumps(validation, indent=2))
