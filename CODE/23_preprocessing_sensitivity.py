# Menguji sensitivitas representasi kategorikal LMS.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import io, itertools, json, os, sys, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
DATA_DIR = PAPER_ROOT / 'DATA'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '07_STABILITY'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

import experiment_lib as el
import care_km as ck

CARE_CONFIG = dict(el.FROZEN_CARE_CONFIG)
K = el.K
BASE_SEED = 4161600
DEGENERATE_MAX_SHARE = 0.95

df_frozen, X_frozen, FEATURES = el.load_lms_dataset(PAPER_ROOT)
df_frozen = df_frozen.copy()
df_frozen['user'] = df_frozen['user'].astype(str)

SEMANTIC_REFERENCE = PAPER_ROOT / 'OUTPUT' / '06_FSLSM_PROFILES' / '08_theory_guided_pseudolabeled_dataset.csv'
EVIDENCE_FILE = PAPER_ROOT / 'OUTPUT' / '06_FSLSM_PROFILES' / '08_semantic_evidence_matrix.csv'

if not SEMANTIC_REFERENCE.is_file() or not EVIDENCE_FILE.is_file():
    raise FileNotFoundError(
        'Output 08_fslsm_semantic_interpretation.py belum lengkap. Jalankan 08_fslsm_semantic_interpretation.py.'
    )

semantic_reference = pd.read_csv(SEMANTIC_REFERENCE, dtype={'user':str})
evidence = pd.read_csv(EVIDENCE_FILE)

semantic_reference = (
    semantic_reference.set_index('user')
    .loc[df_frozen['user']]
    .reset_index()
)

print('PAPER_ROOT:', PAPER_ROOT)
print('Rows:', len(df_frozen))
print('Features:', len(FEATURES))

FEATURE_PLAN = [
    {'feature':'content_visit','threshold_mode':'GRAF_FIXED','graf_low':75.0,'graf_high':100.0},
    {'feature':'content_stay','threshold_mode':'EMPIRICAL_POSITIVE_Q33_Q67','graf_low':50.0,'graf_high':75.0},
    {'feature':'selfass_visit_proxy','threshold_mode':'GRAF_FIXED','graf_low':25.0,'graf_high':75.0},
    {'feature':'selfass_stay_proxy','threshold_mode':'EMPIRICAL_POSITIVE_Q33_Q67','graf_low':50.0,'graf_high':75.0},
    {'feature':'exercise_visit_proxy','threshold_mode':'GRAF_FIXED','graf_low':25.0,'graf_high':75.0},
    {'feature':'exercise_stay_proxy','threshold_mode':'EMPIRICAL_POSITIVE_Q33_Q67','graf_low':50.0,'graf_high':75.0},
    {'feature':'forum_visit','threshold_mode':'GRAF_WITH_DEGENERACY_FALLBACK','graf_low':7.0,'graf_high':50.0},
    {'feature':'forum_stay','threshold_mode':'GRAF_WITH_DEGENERACY_FALLBACK','graf_low':5.0,'graf_high':30.0},
    {'feature':'forum_post','threshold_mode':'GRAF_WITH_SPARSE_FALLBACK','graf_low':1.0,'graf_high':10.0},
    {'feature':'quiz_stay_results_proxy','threshold_mode':'GRAF_FIXED','graf_low':30.0,'graf_high':60.0},
    {'feature':'navigation_overview_visit','threshold_mode':'GRAF_WITH_DEGENERACY_FALLBACK','graf_low':10.0,'graf_high':20.0},
    {'feature':'navigation_overview_stay','threshold_mode':'EMPIRICAL_POSITIVE_Q33_Q67','graf_low':50.0,'graf_high':75.0},
    {'feature':'return_to_overview_rate','threshold_mode':'EMPIRICAL_POSITIVE_Q33_Q67','graf_low':np.nan,'graf_high':np.nan},
]

if [x['feature'] for x in FEATURE_PLAN] != FEATURES:
    raise RuntimeError('Feature plan tidak sama dengan frozen feature order.')


numeric_file = DATA_DIR / 'MAIN_LMS' / 'main_lms_numeric.csv'
if not numeric_file.is_file():
    raise FileNotFoundError(numeric_file)
numeric = pd.read_csv(numeric_file, dtype={'user':str})[['user'] + FEATURES].copy()
for fitur in FEATURES:
    numeric[fitur] = pd.to_numeric(numeric[fitur], errors='coerce').fillna(0.0)
NUMERIC_SOURCE = str(numeric_file)
print('Numeric source:', NUMERIC_SOURCE)

def classify_two_cut(series, low, high):
    x = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(float)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f'Invalid threshold {low}, {high}')
    return pd.Series(
        np.where(x < low, 'L', np.where(x < high, 'M', 'H')),
        index=series.index,
        dtype='object'
    )


def category_share(cat):
    return cat.value_counts(normalize=True).reindex(['L','M','H'], fill_value=0.0)


def positive_quantile_thresholds(series, q1, q2):
    x = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(float)
    pos = x[x > 0]
    if pos.empty:
        return np.nan, np.nan

    low = float(pos.quantile(q1))
    high = float(pos.quantile(q2))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        uniq = np.sort(pos.unique())
        if len(uniq) < 2:
            return float(uniq[0]), float(uniq[0])
        low = float(np.quantile(uniq, q1))
        high = float(np.quantile(uniq, q2))

    return low, high


def sparse_zero_median_categories(series):
    x = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(float)
    pos = x[x > 0]
    if pos.empty:
        return pd.Series('L', index=series.index, dtype='object'), 0.0, np.nan
    med = float(pos.median())
    cat = pd.Series(
        np.where(x <= 0, 'L', np.where(x < med, 'M', 'H')),
        index=series.index,
        dtype='object'
    )
    return cat, 0.0, med


def build_categories(numeric_df, q1=1/3, q2=2/3, graf_scale=1.0):
    cat = pd.DataFrame({'user':numeric_df['user'].astype(str)})
    audit = []

    for spec in FEATURE_PLAN:
        f = spec['feature']
        mode = spec['threshold_mode']
        x = numeric_df[f]

        gl = spec['graf_low']
        gh = spec['graf_high']
        gl = float(gl * graf_scale) if np.isfinite(gl) else np.nan
        gh = float(gh * graf_scale) if np.isfinite(gh) else np.nan

        low = high = np.nan
        fallback = False

        if mode == 'GRAF_FIXED':
            c = classify_two_cut(x, gl, gh)
            low, high = gl, gh
            applied = 'GRAF_FIXED'

        elif mode == 'EMPIRICAL_POSITIVE_Q33_Q67':
            low, high = positive_quantile_thresholds(x, q1, q2)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise RuntimeError(f'Quantile threshold gagal untuk {f}.')
            c = classify_two_cut(x, low, high)
            applied = f'POSITIVE_Q{q1:.4f}_Q{q2:.4f}'

        elif mode == 'GRAF_WITH_DEGENERACY_FALLBACK':
            c0 = classify_two_cut(x, gl, gh)
            sh = category_share(c0)
            deg = float(sh.max()) > DEGENERATE_MAX_SHARE or int((sh == 0).sum()) > 0
            if deg:
                low, high = positive_quantile_thresholds(x, q1, q2)
                if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                    raise RuntimeError(f'Fallback gagal untuk {f}.')
                c = classify_two_cut(x, low, high)
                applied = f'GRAF_FALLBACK_POSITIVE_Q{q1:.4f}_Q{q2:.4f}'
                fallback = True
            else:
                c = c0
                low, high = gl, gh
                applied = 'GRAF_FIXED_TRANSFERABLE'

        elif mode == 'GRAF_WITH_SPARSE_FALLBACK':
            c0 = classify_two_cut(x, gl, gh)
            sh = category_share(c0)
            deg = float(sh.max()) > DEGENERATE_MAX_SHARE or int((sh == 0).sum()) > 0
            if deg:
                c, low, high = sparse_zero_median_categories(x)
                applied = 'GRAF_FALLBACK_ZERO_AWARE_POSITIVE_MEDIAN'
                fallback = True
            else:
                c = c0
                low, high = gl, gh
                applied = 'GRAF_FIXED_TRANSFERABLE'
        else:
            raise ValueError(mode)

        cat[f] = c
        sh = category_share(c)
        audit.append({
            'feature':f,
            'applied_method':applied,
            'applied_low':low,
            'applied_high':high,
            'fallback_used':fallback,
            'L_share':float(sh['L']),
            'M_share':float(sh['M']),
            'H_share':float(sh['H']),
            'n_categories':int(c.nunique()),
        })

    scaled = pd.DataFrame({'user':cat['user']})
    for f in FEATURES:
        scaled[f] = cat[f].map({'L':'1','M':'2','H':'3'}).astype(str)

    return scaled, pd.DataFrame(audit)


CONFIGS = [
    {'name':'PRIMARY','q1':1/3,'q2':2/3,'graf_scale':1.00},
    {'name':'EMPIRICAL_Q30_Q70','q1':0.30,'q2':0.70,'graf_scale':1.00},
    {'name':'EMPIRICAL_Q35_Q65','q1':0.35,'q2':0.65,'graf_scale':1.00},
    {'name':'GRAF_MINUS_10PCT','q1':1/3,'q2':2/3,'graf_scale':0.90},
    {'name':'GRAF_PLUS_10PCT','q1':1/3,'q2':2/3,'graf_scale':1.10},
]

primary_rebuilt, _ = build_categories(numeric)
primary_X = primary_rebuilt[FEATURES].to_numpy(dtype=object)

mismatch = int((primary_X != X_frozen).sum())
if mismatch:
    raise RuntimeError(
        f'PRIMARY reconstruction tidak identik dengan frozen dataset: {mismatch} cell berbeda.'
    )

print('PRIMARY reconstruction: EXACT MATCH')

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
DIM_KEYS = [
    'active_reflective',
    'sensing_intuitive',
    'visual_verbal',
    'sequential_global',
]
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


def semantic_mapping(X, modes, labels):
    X = np.asarray(X, dtype=object)
    labels = np.asarray(labels, dtype=int)
    modes = np.asarray(modes, dtype=object)
    fpos = {f:i for i,f in enumerate(FEATURES)}
    rows = []

    for cid in range(K):
        selected, abbrs = [], []

        for dimension, (first, second) in DIMENSIONS:
            rules = evidence[evidence['dimension'].astype(str).eq(dimension)]

            mv = np.asarray([
                signed_vote(modes[cid, fpos[str(r.feature)]], r)
                for r in rules.itertuples(index=False)
            ])
            ms = int(np.sum(mv > 0) - np.sum(mv < 0))

            member_X = X[labels == cid]
            nv = []
            for r in rules.itertuples(index=False):
                j = fpos[str(r.feature)]
                nv.extend(signed_vote(v, r) for v in member_X[:, j])
            nv = np.asarray(nv)
            ns = int(np.sum(nv > 0) - np.sum(nv < 0))

            if ms > 0:
                pole = first
            elif ms < 0:
                pole = second
            elif ns > 0:
                pole = first
            elif ns < 0:
                pole = second
            else:
                pole = None

            selected.append(pole)
            abbrs.append('Balanced' if pole is None else ABBR[pole])

        unresolved = any(x is None for x in selected)
        rows.append({
            'cluster_id':cid,
            'active_reflective':abbrs[0],
            'sensing_intuitive':abbrs[1],
            'visual_verbal':abbrs[2],
            'sequential_global':abbrs[3],
            'ls_class':'AMBIGUOUS' if unresolved else LS_LOOKUP[tuple(selected)],
            'mapping_status':'BALANCED' if unresolved else 'RESOLVED',
        })

    return pd.DataFrame(rows)

summary_rows = []
threshold_frames = []
feature_change_rows = []
learner_frames = []

primary_labels = None
primary_categorical = None

for cfg in CONFIGS:
    name = cfg['name']
    print('\n===', name, '===')

    cat_df, audit = build_categories(
        numeric,
        q1=cfg['q1'],
        q2=cfg['q2'],
        graf_scale=cfg['graf_scale'],
    )
    audit.insert(0, 'configuration', name)
    threshold_frames.append(audit)

    X = cat_df[FEATURES].to_numpy(dtype=object)

    row, _, result = ck.run_method(
        'CARE-KM', X, None, K,
        seed=BASE_SEED,
        care_config=CARE_CONFIG,
        si_sample_size=el.SI_SAMPLE_SIZE,
        si_seed=BASE_SEED,
        max_iter=el.MAX_ITER,
    )
    labels = np.asarray(result['labels'], dtype=int)

    if name == 'PRIMARY':
        primary_labels = labels.copy()
        primary_categorical = X.copy()

        ref_file = PAPER_ROOT / 'OUTPUT' / '01_FULL_LMS' / '04_carekm_reference_labels.csv'
        if ref_file.is_file():
            ref = pd.read_csv(ref_file, dtype={'user':str})
            rm = ref.set_index('user')['cluster_id'].astype(int)
            ref_labels = df_frozen['user'].map(rm).to_numpy()
            if adjusted_rand_score(ref_labels, labels) < 1 - 1e-12:
                raise RuntimeError('PRIMARY partition berbeda dari 04_lms_full_dataset_experiment.py reference.')

    mapping = semantic_mapping(X, result['modes'], labels)
    mp = mapping.set_index('cluster_id')

    current = pd.DataFrame({
        'user':df_frozen['user'].astype(str),
        'configuration':name,
        'cluster_id':labels,
    })
    for c in DIM_KEYS + ['ls_class','mapping_status']:
        current[c] = [mp.loc[int(k), c] for k in labels]

    ref = semantic_reference[['user','ls_class',*DIM_KEYS]].copy()
    ref = ref.rename(columns={c:f'{c}_reference' for c in ['ls_class',*DIM_KEYS]})
    current = current.merge(ref, on='user', how='left', validate='one_to_one')

    for c in DIM_KEYS:
        current[f'{c}_agreement'] = (
            current[c].astype(str) == current[f'{c}_reference'].astype(str)
        )

    current['mean_dimension_agreement'] = current[
        [f'{c}_agreement' for c in DIM_KEYS]
    ].mean(axis=1)
    current['base_ls_agreement'] = (
        current['ls_class'].astype(str) == current['ls_class_reference'].astype(str)
    )
    learner_frames.append(current)

    changed = X != primary_categorical
    for j, f in enumerate(FEATURES):
        feature_change_rows.append({
            'configuration':name,
            'feature':f,
            'changed_cells':int(changed[:,j].sum()),
            'changed_share':float(changed[:,j].mean()),
        })

    summary_rows.append({
        'configuration':name,
        'q1':cfg['q1'],
        'q2':cfg['q2'],
        'graf_scale':cfg['graf_scale'],
        'initial_cost':float(row['initial_cost']),
        'final_cost':float(row['final_cost']),
        'dbi':float(row['dbi']),
        'si':float(row['si']),
        'iterations':int(row['iterations']),
        'ari_vs_primary_partition':float(adjusted_rand_score(primary_labels, labels)),
        'nmi_vs_primary_partition':float(normalized_mutual_info_score(primary_labels, labels)),
        'changed_categorical_cell_share':float(changed.mean()),
        'changed_learner_profile_share':float(np.any(changed, axis=1).mean()),
        'active_reflective_agreement':float(current['active_reflective_agreement'].mean()),
        'sensing_intuitive_agreement':float(current['sensing_intuitive_agreement'].mean()),
        'visual_verbal_agreement':float(current['visual_verbal_agreement'].mean()),
        'sequential_global_agreement':float(current['sequential_global_agreement'].mean()),
        'mean_dimensional_agreement':float(current['mean_dimension_agreement'].mean()),
        'base_ls_agreement':float(current['base_ls_agreement'].mean()),
        'unresolved_learner_rate':float(current['mapping_status'].ne('RESOLVED').mean()),
    })

summary = pd.DataFrame(summary_rows)
thresholds = pd.concat(threshold_frames, ignore_index=True)
feature_changes = pd.DataFrame(feature_change_rows)
learner_semantic = pd.concat(learner_frames, ignore_index=True)

el.save_csv(summary, OUT_DIR/'preprocessing_sensitivity_summary.csv', index=False)
el.save_csv(thresholds, OUT_DIR/'preprocessing_sensitivity_thresholds.csv', index=False)
el.save_csv(feature_changes, OUT_DIR/'preprocessing_sensitivity_feature_changes.csv', index=False)
el.save_csv(learner_semantic, OUT_DIR/'preprocessing_sensitivity_learner_semantic.csv', index=False)

print(summary)

validation = {
    'analysis':'preprocessing sensitivity',
    'primary_reconstruction_exact_match':True,
    'numeric_source':'DATA/MAIN_LMS/main_lms_numeric.csv',
    'configurations':CONFIGS,
    'tuning_or_model_selection':False,
    'care_config':CARE_CONFIG,
    'K':K,
    'semantic_rule':'mode-first; member evidence only for exact mode tie',
    'outputs':[
        'preprocessing_sensitivity_summary.csv',
        'preprocessing_sensitivity_thresholds.csv',
        'preprocessing_sensitivity_feature_changes.csv',
        'preprocessing_sensitivity_learner_semantic.csv',
    ],
}
(OUT_DIR/'preprocessing_sensitivity_validation.json').write_text(
    json.dumps(validation, indent=2), encoding='utf-8'
)
print(json.dumps(validation, indent=2))
