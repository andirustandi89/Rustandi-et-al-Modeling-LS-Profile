# Menguji sensitivitas aturan konsolidasi semantic profile.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import itertools, json, os, sys
import numpy as np
import pandas as pd


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '07_STABILITY'
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el

K = el.K
df, X, FEATURES = el.load_lms_dataset(PAPER_ROOT)
df = df.copy()
df['user'] = df['user'].astype(str)

MODE_FILE = PAPER_ROOT/'OUTPUT'/'01_FULL_LMS'/'04_carekm_reference_final_modes.csv'
LABEL_FILE = PAPER_ROOT/'OUTPUT'/'01_FULL_LMS'/'04_carekm_reference_labels.csv'
EVIDENCE_FILE = PAPER_ROOT/'OUTPUT'/'06_FSLSM_PROFILES'/'08_semantic_evidence_matrix.csv'
PRIMARY_MAP_FILE = PAPER_ROOT/'OUTPUT'/'06_FSLSM_PROFILES'/'08_cluster_to_fslsm_mapping.csv'

for p in [MODE_FILE, LABEL_FILE, EVIDENCE_FILE, PRIMARY_MAP_FILE]:
    if not p.is_file():
        raise FileNotFoundError(
            f'{p.name} tidak ditemukan. Jalankan 04_lms_full_dataset_experiment.py dan 08 terlebih dahulu.'
        )

mode_df = pd.read_csv(MODE_FILE, dtype=str)
label_df = pd.read_csv(LABEL_FILE, dtype={'user':str})
evidence = pd.read_csv(EVIDENCE_FILE)
primary_file_mapping = pd.read_csv(PRIMARY_MAP_FILE)

modes = mode_df[FEATURES].astype(str).to_numpy(dtype=object)
label_map = label_df.set_index('user')['cluster_id'].astype(int)
labels = df['user'].map(label_map).to_numpy(dtype=int)

if modes.shape != (K, len(FEATURES)):
    raise RuntimeError('Reference mode shape tidak valid.')

cluster_sizes = np.bincount(labels, minlength=K)
fpos = {f:i for i,f in enumerate(FEATURES)}

print('PAPER_ROOT:', PAPER_ROOT)
print('Rows:', len(df))
print('K:', K)

DIMENSIONS = [
    ('Active/Reflective', ('Active','Reflective')),
    ('Sensing/Intuitive', ('Sensing','Intuitive')),
    ('Visual/Verbal', ('Visual','Verbal')),
    ('Sequential/Global', ('Sequential','Global')),
]
POLES = {d:p for d,p in DIMENSIONS}
ABBR = {
    'Active':'Act.','Reflective':'Ref.',
    'Sensing':'Sens.','Intuitive':'Int.',
    'Visual':'Vis.','Verbal':'Ver.',
    'Sequential':'Seq.','Global':'Gl.'
}
DIM_KEYS = {
    'Active/Reflective':'active_reflective',
    'Sensing/Intuitive':'sensing_intuitive',
    'Visual/Verbal':'visual_verbal',
    'Sequential/Global':'sequential_global',
}
LS_LOOKUP = {
    tuple(combo):f'LS{i:02d}'
    for i, combo in enumerate(
        itertools.product(*[p for _,p in DIMENSIONS]), start=1
    )
}


def signed_vote(value, rule):
    value = int(value)
    first, second = POLES[str(rule.dimension)]
    if value == 2:
        return 0
    pole = str(rule.high_supports) if value == 3 else str(rule.low_supports)
    if pole == first:
        return 1
    if pole == second:
        return -1
    raise ValueError(pole)


def evidence_scores(cluster_id, dimension):
    rules = evidence[evidence['dimension'].astype(str).eq(dimension)]
    first, second = POLES[dimension]

    mv = np.asarray([
        signed_vote(modes[cluster_id, fpos[str(r.feature)]], r)
        for r in rules.itertuples(index=False)
    ])
    mode_score = int(np.sum(mv > 0) - np.sum(mv < 0))

    member_X = X[labels == cluster_id]
    nv = []
    for r in rules.itertuples(index=False):
        j = fpos[str(r.feature)]
        nv.extend(signed_vote(v, r) for v in member_X[:,j])
    nv = np.asarray(nv)
    member_score = int(np.sum(nv > 0) - np.sum(nv < 0))

    return {
        'first':first,
        'second':second,
        'mode_score':mode_score,
        'member_score':member_score,
    }


def select_pole(scores, rule):
    first, second = scores['first'], scores['second']
    ms, ns = scores['mode_score'], scores['member_score']

    if rule == 'PRIMARY_MODE_FIRST':
        if ms > 0: return first, 'MODE'
        if ms < 0: return second, 'MODE'
        if ns > 0: return first, 'MEMBER'
        if ns < 0: return second, 'MEMBER'
        return None, 'BALANCED'

    if rule == 'MEMBER_FIRST':
        if ns > 0: return first, 'MEMBER'
        if ns < 0: return second, 'MEMBER'
        if ms > 0: return first, 'MODE'
        if ms < 0: return second, 'MODE'
        return None, 'BALANCED'

    if rule == 'MODE_ONLY':
        if ms > 0: return first, 'MODE'
        if ms < 0: return second, 'MODE'
        return None, 'BALANCED'

    if rule == 'MEMBER_ONLY':
        if ns > 0: return first, 'MEMBER'
        if ns < 0: return second, 'MEMBER'
        return None, 'BALANCED'

    raise ValueError(rule)


def build_mapping(rule):
    rows = []

    for cid in range(K):
        selected = []
        abbrs = []

        row = {
            'rule':rule,
            'cluster_id':cid,
            'cluster_name':f'C{cid+1:02d}',
            'cluster_size':int(cluster_sizes[cid]),
        }

        for dimension, _ in DIMENSIONS:
            scores = evidence_scores(cid, dimension)
            pole, source = select_pole(scores, rule)
            key = DIM_KEYS[dimension]

            selected.append(pole)
            abbrs.append('Balanced' if pole is None else ABBR[pole])

            row[key] = abbrs[-1]
            row[f'{key}_source'] = source
            row[f'{key}_mode_score'] = scores['mode_score']
            row[f'{key}_member_score'] = scores['member_score']

        unresolved = any(x is None for x in selected)
        row['semantic_label'] = '-'.join(abbrs)
        row['ls_class'] = 'AMBIGUOUS' if unresolved else LS_LOOKUP[tuple(selected)]
        row['mapping_status'] = 'BALANCED' if unresolved else 'RESOLVED'
        row['unresolved_dimensions'] = int(sum(x is None for x in selected))
        rows.append(row)

    return pd.DataFrame(rows)


RULES = [
    'PRIMARY_MODE_FIRST',
    'MEMBER_FIRST',
    'MODE_ONLY',
    'MEMBER_ONLY',
]

mappings = pd.concat([build_mapping(r) for r in RULES], ignore_index=True)
primary = mappings[mappings['rule'].eq('PRIMARY_MODE_FIRST')].copy()

check_cols = [
    'active_reflective',
    'sensing_intuitive',
    'visual_verbal',
    'sequential_global',
    'ls_class',
    'mapping_status',
]

expected = (
    primary_file_mapping.set_index('cluster_id')[check_cols]
    .astype(str)
    .sort_index()
)
observed = (
    primary.set_index('cluster_id')[check_cols]
    .astype(str)
    .sort_index()
)

if not observed.equals(expected):
    n_diff = int((observed != expected).to_numpy().sum())
    raise RuntimeError(
        f'PRIMARY semantic rule tidak identik dengan 08_fslsm_semantic_interpretation.py: {n_diff} perbedaan.'
    )

print('PRIMARY semantic rule: EXACT MATCH with 08_fslsm_semantic_interpretation.py')

primary_idx = primary.set_index('cluster_id')
summary_rows = []
dimension_rows = []
comparison_rows = []

for rule in RULES:
    cur = mappings[mappings['rule'].eq(rule)].copy()
    cur_idx = cur.set_index('cluster_id')

    weights = primary_idx['cluster_size'].astype(float)
    weighted_agreements = []
    cluster_agreements = []

    for dimension, _ in DIMENSIONS:
        key = DIM_KEYS[dimension]
        match = cur_idx[key].astype(str) == primary_idx[key].astype(str)

        weighted = float(np.average(match.astype(float), weights=weights))
        unweighted = float(match.mean())

        weighted_agreements.append(weighted)
        cluster_agreements.append(unweighted)

        dimension_rows.append({
            'rule':rule,
            'dimension':dimension,
            'cluster_level_agreement':unweighted,
            'learner_weighted_agreement':weighted,
            'changed_clusters':int((~match).sum()),
        })

    profile_match = (
        cur_idx['ls_class'].astype(str)
        == primary_idx['ls_class'].astype(str)
    )

    unresolved_mask = cur_idx['mapping_status'].astype(str).ne('RESOLVED')
    unresolved_clusters = int(unresolved_mask.sum())
    unresolved_learners = int(cur_idx.loc[unresolved_mask, 'cluster_size'].sum())

    summary_rows.append({
        'rule':rule,
        'active_reflective_agreement':weighted_agreements[0],
        'sensing_intuitive_agreement':weighted_agreements[1],
        'visual_verbal_agreement':weighted_agreements[2],
        'sequential_global_agreement':weighted_agreements[3],
        'mean_dimensional_agreement_weighted':float(np.mean(weighted_agreements)),
        'mean_dimensional_agreement_cluster_level':float(np.mean(cluster_agreements)),
        'exact_base_ls_agreement_weighted':float(
            np.average(profile_match.astype(float), weights=weights)
        ),
        'exact_base_ls_agreement_cluster_level':float(profile_match.mean()),
        'unresolved_clusters':unresolved_clusters,
        'unresolved_learners':unresolved_learners,
        'unresolved_learner_rate':float(unresolved_learners / len(df)),
        'distinct_resolved_ls_classes':int(
            cur_idx.loc[~unresolved_mask, 'ls_class'].nunique()
        ),
    })

    for cid in range(K):
        row = {
            'rule':rule,
            'cluster_id':cid,
            'cluster_name':f'C{cid+1:02d}',
            'cluster_size':int(primary_idx.loc[cid,'cluster_size']),
            'primary_ls_class':primary_idx.loc[cid,'ls_class'],
            'alternative_ls_class':cur_idx.loc[cid,'ls_class'],
            'base_ls_match':bool(profile_match.loc[cid]),
        }
        for dimension, _ in DIMENSIONS:
            key = DIM_KEYS[dimension]
            row[f'{key}_primary'] = primary_idx.loc[cid,key]
            row[f'{key}_alternative'] = cur_idx.loc[cid,key]
            row[f'{key}_match'] = bool(
                str(primary_idx.loc[cid,key]) == str(cur_idx.loc[cid,key])
            )
        comparison_rows.append(row)

summary = pd.DataFrame(summary_rows)
dimension_summary = pd.DataFrame(dimension_rows)
cluster_comparison = pd.DataFrame(comparison_rows)

el.save_csv(
    mappings,
    OUT_DIR/'semantic_rule_sensitivity_all_mappings.csv',
    index=False
)
el.save_csv(
    summary,
    OUT_DIR/'semantic_rule_sensitivity_summary.csv',
    index=False
)
el.save_csv(
    dimension_summary,
    OUT_DIR/'semantic_rule_sensitivity_dimension_summary.csv',
    index=False
)
el.save_csv(
    cluster_comparison,
    OUT_DIR/'semantic_rule_sensitivity_cluster_comparison.csv',
    index=False
)

print(summary)
print(dimension_summary)

validation = {
    'analysis':'semantic rule sensitivity',
    'clustering_changed':False,
    'reference_partition':'04_carekm_reference_labels.csv',
    'semantic_evidence_changed':False,
    'reference_mapping':'08_cluster_to_fslsm_mapping.csv',
    'primary_reproduced_exactly':True,
    'hungarian_assignment':False,
    'one_to_one_constraint':False,
    'rules':{
        'PRIMARY_MODE_FIRST':'mode direction; member evidence only for exact mode tie',
        'MEMBER_FIRST':'member direction; mode evidence only for exact member tie',
        'MODE_ONLY':'mode direction only; exact mode ties unresolved',
        'MEMBER_ONLY':'member direction only; exact member ties unresolved',
    },
    'outputs':[
        'semantic_rule_sensitivity_all_mappings.csv',
        'semantic_rule_sensitivity_summary.csv',
        'semantic_rule_sensitivity_dimension_summary.csv',
        'semantic_rule_sensitivity_cluster_comparison.csv',
    ],
}
(OUT_DIR/'semantic_rule_sensitivity_validation.json').write_text(
    json.dumps(validation, indent=2), encoding='utf-8'
)
print(json.dumps(validation, indent=2))
