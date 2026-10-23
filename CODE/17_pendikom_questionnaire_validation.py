# Menguji correspondence terhadap ILS pada cohort PENDIKOM.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import os, sys, json, csv, itertools, warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


PROJECT_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

DATA_DIR = PROJECT_ROOT/'DATA'
LIB_DIR = PROJECT_ROOT/'CODE'/'LIB'
OUTPUT_DIR = PROJECT_ROOT / 'OUTPUT' / '09_QUESTIONNAIRE_HOLDOUT'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

import care_km as ck

GROUNDTRUTH_CSV = DATA_DIR/'PENDIKOM'/'compiled_pendikom_moodle_logs.csv'
GROUNDTRUTH_SPLIT = DATA_DIR/'PENDIKOM'/'pendikom_ils_split_161600.csv'

SPLIT_SEED = 161600
TEST_SIZE = 0.30
K = 16
MAX_ITER = 100
CARE_CONFIG = dict(
    candidate_rule='separation',
    candidate_budget=16,
    min_gain=0.0,
)

DIM_NAMES = [
    'Active / Reflective',
    'Sensing / Intuitive',
    'Visual / Verbal',
    'Sequential / Global',
]
DIM_KEYS = [
    'active_reflective',
    'sensing_intuitive',
    'visual_verbal',
    'sequential_global',
]

print('PROJECT_ROOT   :', PROJECT_ROOT)
print('GROUNDTRUTH_CSV:', GROUNDTRUTH_CSV)
print('SPLIT_FILE     :', GROUNDTRUTH_SPLIT)
print('OUTPUT_DIR     :', OUTPUT_DIR)

DERIVED_REFINEMENT_BUDGET = ck.derive_refinement_budget(CARE_CONFIG['candidate_budget'])
assert DERIVED_REFINEMENT_BUDGET == 4
PIPELINE_VERSION = 'carekm-questionnaire-v6'
print('Versi pipeline questionnaire:', PIPELINE_VERSION)
print('CARE implementation:', ck.CARE_IMPLEMENTATION_VERSION)
print('Derived refinement budget:', DERIVED_REFINEMENT_BUDGET)


POLES = {
    'Active/Reflective': ('Active','Reflective'),
    'Sensing/Intuitive': ('Sensing','Intuitive'),
    'Visual/Verbal': ('Visual','Verbal'),
    'Sequential/Global': ('Sequential','Global'),
}
ABBR = {
    'Active':'Act.','Reflective':'Ref.','Sensing':'Sens.','Intuitive':'Int.',
    'Visual':'Vis.','Verbal':'Ver.','Sequential':'Seq.','Global':'Gl.'
}
DIMENSION_ORDER = list(POLES.keys())
LS_LOOKUP = {
    tuple(combo): f'LS{i:02d}'
    for i, combo in enumerate(
        itertools.product(*[POLES[d] for d in DIMENSION_ORDER]),
        start=1,
    )
}
LS_TO_POLES = {v:k for k,v in LS_LOOKUP.items()}

def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=True)

def categorical_nb():
    try:
        return CategoricalNB(alpha=1.0, min_categories=4)
    except TypeError:
        return CategoricalNB(alpha=1.0)

def build_external_classifiers():
    return {
        'NB': categorical_nb(),
        'DT': DecisionTreeClassifier(
            random_state=SPLIT_SEED, class_weight='balanced'
        ),
        'SVM': Pipeline([
            ('onehot', one_hot_encoder()),
            ('svc', SVC(
                kernel='rbf', C=1.0, gamma='scale',
                class_weight='balanced'
            )),
        ]),
    }

def fit_care(X):
    X = np.asarray(X, dtype=object)
    init = ck.care_init(X, K, **CARE_CONFIG)
    result = ck.kmodes_fit(X, init, max_iter=MAX_ITER)
    return init, result

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
    raise ValueError(f'Unknown semantic pole: {pole}')

def semantic_mapping(X, labels, modes, features, semantic_rules):
    X = np.asarray(X, dtype=object)
    labels = np.asarray(labels, dtype=int)
    modes = np.asarray(modes, dtype=object)
    fpos = {f:i for i,f in enumerate(features)}
    rows = []

    for cid in range(K):
        selected = []
        short = []

        for dim in DIMENSION_ORDER:
            rules = semantic_rules[
                semantic_rules['dimension'].astype(str).eq(dim)
            ]
            mode_votes = np.asarray([
                signed_vote(modes[cid, fpos[str(r.feature)]], r)
                for r in rules.itertuples(index=False)
            ], dtype=int)
            mode_score = int(
                np.sum(mode_votes > 0) - np.sum(mode_votes < 0)
            )

            member_X = X[labels == cid]
            member_votes = []
            for r in rules.itertuples(index=False):
                j = fpos[str(r.feature)]
                member_votes.extend(
                    signed_vote(v, r) for v in member_X[:, j]
                )
            member_votes = np.asarray(member_votes, dtype=int)
            member_score = (
                int(np.sum(member_votes > 0) - np.sum(member_votes < 0))
                if len(member_votes) else 0
            )

            first, second = POLES[dim]
            pole = (
                first if mode_score > 0 else
                second if mode_score < 0 else
                first if member_score > 0 else
                second if member_score < 0 else
                None
            )
            selected.append(pole)
            short.append('Balanced' if pole is None else ABBR[pole])

        unresolved = any(x is None for x in selected)
        rows.append({
            'cluster_id': cid,
            'cluster_name': f'C{cid+1:02d}',
            'cluster_size': int(np.sum(labels == cid)),
            'active_reflective': short[0],
            'sensing_intuitive': short[1],
            'visual_verbal': short[2],
            'sequential_global': short[3],
            'ls_class': (
                'AMBIGUOUS'
                if unresolved
                else LS_LOOKUP[tuple(selected)]
            ),
            'mapping_status': 'BALANCED' if unresolved else 'RESOLVED',
        })

    return pd.DataFrame(rows)

def pole_abbr_to_full(x):
    return {
        'Act.':'Active','Ref.':'Reflective',
        'Sens.':'Sensing','Int.':'Intuitive',
        'Vis.':'Visual','Ver.':'Verbal',
        'Seq.':'Sequential','Gl.':'Global',
    }.get(str(x), 'Balanced')

def ils_code_to_full(code):
    s = str(code)
    for prefix, full in [
        ('Act.','Active'),('Ref.','Reflective'),
        ('Sens.','Sensing'),('Int.','Intuitive'),
        ('Vis.','Visual'),('Ver.','Verbal'),
        ('Seq.','Sequential'),('Gl.','Global'),
    ]:
        if s.startswith(prefix):
            return full
    raise ValueError(f'Unexpected ILS code: {code}')

def save_confusion(y_true, y_pred, path):
    labels = sorted(set(map(str,y_true)) | set(map(str,y_pred)))
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(path)


PENDIKOM_FEATURES = [
    'content_visit','content_stay','selfass_visit','selfass_twice_wrong',
    'exercise_visit','exercise_stay','ques_detail','ques_overview',
    'ques_facts','ques_concepts','ques_graphics','ques_text',
    'ques_interpret','ques_develop','quiz_revisions','quiz_stay_results',
    'navigation_skip','navigation_overview_visit','navigation_overview_stay',
]

PENDIKOM_THRESHOLDS = {
    'content_visit': (75,100),
    'content_stay': (50,75),
    'selfass_visit': (25,75),
    'selfass_twice_wrong': (25,50),
    'exercise_visit': (25,75),
    'exercise_stay': (50,75),
    'ques_detail': (50,75),
    'ques_overview': (50,75),
    'ques_facts': (50,75),
    'ques_concepts': (50,75),
    'ques_graphics': (50,75),
    'ques_text': (50,75),
    'ques_interpret': (50,75),
    'ques_develop': (50,75),
    'quiz_revisions': (20,50),
    'quiz_stay_results': (30,60),
    'navigation_skip': (1,2),
    'navigation_overview_visit': (10,20),
    'navigation_overview_stay': (50,75),
}

ILS_CODE_COLS = [
    'ils_active_reflective_code',
    'ils_sensing_intuitive_code',
    'ils_visual_verbal_code',
    'ils_sequential_global_code',
]


def read_compiled_pendikom(path):
    """Membaca CSV normal atau format row-wrapped satu kolom dari sumber publik."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        outer = list(csv.reader(fh))

    if outer and all(len(r) == 1 for r in outer):
        inner = [next(csv.reader([r[0]])) for r in outer]
        width = len(inner[0])
        bad = [i for i, r in enumerate(inner) if len(r) != width]
        if bad:
            raise RuntimeError(f'Wrapped CSV has inconsistent row widths: {bad[:10]}')
        frame = pd.DataFrame(inner[1:], columns=inner[0])
    else:
        frame = pd.read_csv(path, low_memory=False)

    frame.columns = [str(c).strip() for c in frame.columns]

    text_cols = {
        'record_group','username_anon','idnumber_anon','course_fullname',
        'ils_course_fullname','ils_active_reflective_code',
        'ils_sensing_intuitive_code','ils_visual_verbal_code',
        'ils_sequential_global_code','ils_ls_class','ils_semantic_label',
    } | {f'Q{i:02d}' for i in range(1,45)}

    for c in frame.columns:
        if c not in text_cols:
            frame[c] = pd.to_numeric(frame[c], errors='coerce')

    return frame


if not GROUNDTRUTH_CSV.is_file():
    raise FileNotFoundError(
        f'{GROUNDTRUTH_CSV} belum tersedia. Jalankan: '
        'python CODE/download_public_data.py --pendikom-only'
    )

pendikom_all = read_compiled_pendikom(GROUNDTRUTH_CSV)

required_source_cols = {
    'has_ils','userid','courseid','total_activity','active_days',
    'content_available','content_visit_evidence_count','content_stay_seconds',
    'question_available','question_attempt_count','repeated_question_count',
    'twice_wrong_question_count','assign_available','submitted_assignment_count',
    'exercise_stay_seconds','quiz_results_stay_seconds_sum',
    'quiz_results_event_count','revised_question_attempt_count',
    'module_view_count','navigation_skip_count','overview_visit_count',
    'overview_stay_seconds','ils_ls_class', *ILS_CODE_COLS,
}
for tag in ['detail','overview','facts','concepts','graphics','text','develop']:
    required_source_cols |= {
        f'ques_{tag}_fraction_sum',
        f'ques_{tag}_attempted_count',
    }
required_source_cols |= {
    'interpret_question_score_sum',
    'ques_interpret_attempted_count',
}

missing = sorted(required_source_cols - set(pendikom_all.columns))
if missing:
    raise RuntimeError(f'PENDIKOM source missing required columns: {missing}')

pendikom_ils = pendikom_all[pendikom_all['has_ils'].eq(1)].copy()
pendikom_ils['userid'] = pd.to_numeric(pendikom_ils['userid'], errors='raise').astype(int)
pendikom_ils['courseid'] = pd.to_numeric(pendikom_ils['courseid'], errors='raise').astype(int)

if pendikom_ils['userid'].duplicated().any():
    raise RuntimeError('Duplicate userid among WITH_ILS learners.')

if len(pendikom_ils) != 461:
    raise RuntimeError(f'Expected 461 WITH_ILS learners, found {len(pendikom_ils)}.')

if pendikom_ils[ILS_CODE_COLS + ['ils_ls_class']].isna().any().any():
    raise RuntimeError('Incomplete ILS ground-truth fields among WITH_ILS learners.')


FROZEN_PENDIKOM_SPLIT_ROWS = [
    (1008, 'train'),
    (1016, 'train'),
    (1026, 'train'),
    (1029, 'train'),
    (1056, 'train'),
    (1059, 'train'),
    (1077, 'train'),
    (1079, 'train'),
    (1103, 'train'),
    (1147, 'train'),
    (1150, 'train'),
    (1164, 'train'),
    (1170, 'train'),
    (1180, 'train'),
    (1183, 'train'),
    (1185, 'train'),
    (1193, 'train'),
    (1196, 'train'),
    (1214, 'train'),
    (1215, 'train'),
    (1224, 'train'),
    (1238, 'train'),
    (1245, 'train'),
    (1252, 'train'),
    (1256, 'train'),
    (1271, 'train'),
    (1284, 'train'),
    (1285, 'train'),
    (1292, 'train'),
    (1315, 'train'),
    (1325, 'train'),
    (1347, 'train'),
    (1349, 'train'),
    (1355, 'train'),
    (1358, 'train'),
    (1361, 'train'),
    (1386, 'train'),
    (1397, 'train'),
    (1403, 'train'),
    (1409, 'train'),
    (1414, 'train'),
    (1421, 'train'),
    (1424, 'train'),
    (1431, 'train'),
    (1435, 'train'),
    (1440, 'train'),
    (1455, 'train'),
    (1461, 'train'),
    (1462, 'train'),
    (1465, 'train'),
    (1473, 'train'),
    (1482, 'train'),
    (1483, 'train'),
    (1505, 'train'),
    (1528, 'train'),
    (1533, 'train'),
    (1535, 'train'),
    (1547, 'train'),
    (1551, 'train'),
    (1564, 'train'),
    (1567, 'train'),
    (1573, 'train'),
    (1592, 'train'),
    (1626, 'train'),
    (1630, 'train'),
    (1634, 'train'),
    (1638, 'train'),
    (1641, 'train'),
    (1648, 'train'),
    (1652, 'train'),
    (1658, 'train'),
    (1673, 'train'),
    (1674, 'train'),
    (1676, 'train'),
    (1685, 'train'),
    (1686, 'train'),
    (1689, 'train'),
    (1700, 'train'),
    (1706, 'train'),
    (1707, 'train'),
    (1718, 'train'),
    (1721, 'train'),
    (1731, 'train'),
    (1735, 'train'),
    (1749, 'train'),
    (1758, 'train'),
    (1762, 'train'),
    (1769, 'train'),
    (1781, 'train'),
    (1784, 'train'),
    (1787, 'train'),
    (1790, 'train'),
    (1791, 'train'),
    (1797, 'train'),
    (1824, 'train'),
    (1825, 'train'),
    (1835, 'train'),
    (1846, 'train'),
    (1850, 'train'),
    (1863, 'train'),
    (1866, 'train'),
    (1867, 'train'),
    (1878, 'train'),
    (1881, 'train'),
    (1890, 'train'),
    (1893, 'train'),
    (1896, 'train'),
    (1906, 'train'),
    (1915, 'train'),
    (1928, 'train'),
    (1932, 'train'),
    (1936, 'train'),
    (1950, 'train'),
    (1960, 'train'),
    (1966, 'train'),
    (1985, 'train'),
    (1994, 'train'),
    (1996, 'train'),
    (2002, 'train'),
    (2010, 'train'),
    (2029, 'train'),
    (2041, 'train'),
    (2049, 'train'),
    (2051, 'train'),
    (2055, 'train'),
    (2060, 'train'),
    (2066, 'train'),
    (2083, 'train'),
    (2088, 'train'),
    (2089, 'train'),
    (2095, 'train'),
    (2096, 'train'),
    (2099, 'train'),
    (2102, 'train'),
    (2106, 'train'),
    (2110, 'train'),
    (2111, 'train'),
    (2116, 'train'),
    (2142, 'train'),
    (2147, 'train'),
    (2149, 'train'),
    (2156, 'train'),
    (2168, 'train'),
    (2169, 'train'),
    (2176, 'train'),
    (2178, 'train'),
    (2189, 'train'),
    (2194, 'train'),
    (2197, 'train'),
    (2208, 'train'),
    (2215, 'train'),
    (2226, 'train'),
    (2227, 'train'),
    (2230, 'train'),
    (2231, 'train'),
    (2254, 'train'),
    (2258, 'train'),
    (2269, 'train'),
    (2301, 'train'),
    (2308, 'train'),
    (2310, 'train'),
    (2311, 'train'),
    (2315, 'train'),
    (2325, 'train'),
    (2343, 'train'),
    (2351, 'train'),
    (2363, 'train'),
    (2365, 'train'),
    (2370, 'train'),
    (2389, 'train'),
    (2405, 'train'),
    (2429, 'train'),
    (2436, 'train'),
    (2461, 'train'),
    (2468, 'train'),
    (2482, 'train'),
    (2493, 'train'),
    (2500, 'train'),
    (2501, 'train'),
    (2503, 'train'),
    (2509, 'train'),
    (2519, 'train'),
    (2531, 'train'),
    (2545, 'train'),
    (2571, 'train'),
    (2575, 'train'),
    (2580, 'train'),
    (2581, 'train'),
    (2583, 'train'),
    (2595, 'train'),
    (2606, 'train'),
    (2609, 'train'),
    (2617, 'train'),
    (2637, 'train'),
    (2638, 'train'),
    (2639, 'train'),
    (2648, 'train'),
    (2652, 'train'),
    (2672, 'train'),
    (2674, 'train'),
    (2677, 'train'),
    (2682, 'train'),
    (2695, 'train'),
    (2707, 'train'),
    (2715, 'train'),
    (2719, 'train'),
    (2728, 'train'),
    (2738, 'train'),
    (2739, 'train'),
    (2742, 'train'),
    (2764, 'train'),
    (2773, 'train'),
    (2785, 'train'),
    (2786, 'train'),
    (2790, 'train'),
    (2793, 'train'),
    (2796, 'train'),
    (2803, 'train'),
    (2829, 'train'),
    (2833, 'train'),
    (2838, 'train'),
    (2864, 'train'),
    (2867, 'train'),
    (2874, 'train'),
    (2885, 'train'),
    (2888, 'train'),
    (2892, 'train'),
    (2896, 'train'),
    (2910, 'train'),
    (2929, 'train'),
    (2946, 'train'),
    (2951, 'train'),
    (2962, 'train'),
    (2969, 'train'),
    (2971, 'train'),
    (2975, 'train'),
    (2979, 'train'),
    (2986, 'train'),
    (2998, 'train'),
    (3004, 'train'),
    (3046, 'train'),
    (3048, 'train'),
    (3054, 'train'),
    (3062, 'train'),
    (3088, 'train'),
    (3095, 'train'),
    (3100, 'train'),
    (3103, 'train'),
    (3111, 'train'),
    (3114, 'train'),
    (3124, 'train'),
    (3135, 'train'),
    (3142, 'train'),
    (3148, 'train'),
    (3150, 'train'),
    (3157, 'train'),
    (3177, 'train'),
    (3182, 'train'),
    (3183, 'train'),
    (3188, 'train'),
    (3195, 'train'),
    (3217, 'train'),
    (3221, 'train'),
    (3224, 'train'),
    (3231, 'train'),
    (3253, 'train'),
    (3262, 'train'),
    (3263, 'train'),
    (3266, 'train'),
    (3271, 'train'),
    (3279, 'train'),
    (3287, 'train'),
    (3291, 'train'),
    (3295, 'train'),
    (3308, 'train'),
    (3320, 'train'),
    (3329, 'train'),
    (3330, 'train'),
    (3333, 'train'),
    (3343, 'train'),
    (3370, 'train'),
    (3374, 'train'),
    (3381, 'train'),
    (3385, 'train'),
    (3407, 'train'),
    (3433, 'train'),
    (3438, 'train'),
    (3456, 'train'),
    (3466, 'train'),
    (3472, 'train'),
    (3476, 'train'),
    (3484, 'train'),
    (3488, 'train'),
    (3492, 'train'),
    (3500, 'train'),
    (3511, 'train'),
    (3519, 'train'),
    (3530, 'train'),
    (3533, 'train'),
    (3541, 'train'),
    (3557, 'train'),
    (3558, 'train'),
    (3562, 'train'),
    (3568, 'train'),
    (3571, 'train'),
    (3579, 'train'),
    (3588, 'train'),
    (3592, 'train'),
    (3608, 'train'),
    (3609, 'train'),
    (3621, 'train'),
    (3630, 'train'),
    (3634, 'train'),
    (3654, 'train'),
    (3658, 'train'),
    (3672, 'train'),
    (3694, 'train'),
    (3706, 'train'),
    (3718, 'train'),
    (3726, 'train'),
    (3744, 'train'),
    (3761, 'train'),
    (1015, 'test'),
    (1023, 'test'),
    (1055, 'test'),
    (1082, 'test'),
    (1107, 'test'),
    (1109, 'test'),
    (1113, 'test'),
    (1115, 'test'),
    (1146, 'test'),
    (1191, 'test'),
    (1204, 'test'),
    (1208, 'test'),
    (1249, 'test'),
    (1273, 'test'),
    (1277, 'test'),
    (1301, 'test'),
    (1314, 'test'),
    (1316, 'test'),
    (1320, 'test'),
    (1339, 'test'),
    (1340, 'test'),
    (1368, 'test'),
    (1383, 'test'),
    (1388, 'test'),
    (1407, 'test'),
    (1430, 'test'),
    (1451, 'test'),
    (1452, 'test'),
    (1475, 'test'),
    (1489, 'test'),
    (1498, 'test'),
    (1508, 'test'),
    (1530, 'test'),
    (1544, 'test'),
    (1562, 'test'),
    (1565, 'test'),
    (1588, 'test'),
    (1657, 'test'),
    (1661, 'test'),
    (1681, 'test'),
    (1712, 'test'),
    (1729, 'test'),
    (1745, 'test'),
    (1752, 'test'),
    (1757, 'test'),
    (1766, 'test'),
    (1809, 'test'),
    (1831, 'test'),
    (1838, 'test'),
    (1845, 'test'),
    (1852, 'test'),
    (1884, 'test'),
    (1934, 'test'),
    (1956, 'test'),
    (1988, 'test'),
    (2000, 'test'),
    (2067, 'test'),
    (2077, 'test'),
    (2084, 'test'),
    (2103, 'test'),
    (2114, 'test'),
    (2126, 'test'),
    (2132, 'test'),
    (2136, 'test'),
    (2160, 'test'),
    (2199, 'test'),
    (2223, 'test'),
    (2242, 'test'),
    (2247, 'test'),
    (2272, 'test'),
    (2275, 'test'),
    (2291, 'test'),
    (2292, 'test'),
    (2299, 'test'),
    (2335, 'test'),
    (2402, 'test'),
    (2415, 'test'),
    (2416, 'test'),
    (2420, 'test'),
    (2472, 'test'),
    (2492, 'test'),
    (2495, 'test'),
    (2577, 'test'),
    (2620, 'test'),
    (2628, 'test'),
    (2632, 'test'),
    (2666, 'test'),
    (2692, 'test'),
    (2704, 'test'),
    (2745, 'test'),
    (2758, 'test'),
    (2789, 'test'),
    (2817, 'test'),
    (2823, 'test'),
    (2881, 'test'),
    (2918, 'test'),
    (2930, 'test'),
    (2932, 'test'),
    (2940, 'test'),
    (2957, 'test'),
    (3016, 'test'),
    (3119, 'test'),
    (3126, 'test'),
    (3144, 'test'),
    (3160, 'test'),
    (3162, 'test'),
    (3165, 'test'),
    (3166, 'test'),
    (3190, 'test'),
    (3196, 'test'),
    (3228, 'test'),
    (3238, 'test'),
    (3240, 'test'),
    (3267, 'test'),
    (3284, 'test'),
    (3303, 'test'),
    (3311, 'test'),
    (3376, 'test'),
    (3392, 'test'),
    (3412, 'test'),
    (3423, 'test'),
    (3447, 'test'),
    (3457, 'test'),
    (3459, 'test'),
    (3463, 'test'),
    (3469, 'test'),
    (3507, 'test'),
    (3548, 'test'),
    (3567, 'test'),
    (3585, 'test'),
    (3614, 'test'),
    (3647, 'test'),
    (3668, 'test'),
    (3678, 'test'),
    (3689, 'test'),
    (3690, 'test'),
    (3720, 'test'),
    (3728, 'test'),
    (3742, 'test')
]

if GROUNDTRUTH_SPLIT.is_file():
    split = pd.read_csv(GROUNDTRUTH_SPLIT)
    print('Using frozen split file:', GROUNDTRUTH_SPLIT)
else:
    split = pd.DataFrame(
        FROZEN_PENDIKOM_SPLIT_ROWS,
        columns=['userid', 'split']
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(GROUNDTRUTH_SPLIT, index=False)
    print('Frozen split file was missing.')
    print('Reconstructed and saved exact split to:', GROUNDTRUTH_SPLIT)
if not {'userid','split'}.issubset(split.columns):
    raise RuntimeError('Frozen split must contain userid and split columns.')

split['userid'] = pd.to_numeric(split['userid'], errors='raise').astype(int)
split['split'] = split['split'].astype(str).str.strip().str.lower()

if split['userid'].duplicated().any():
    raise RuntimeError('Frozen split contains duplicate userid values.')
if set(split['split']) != {'train','test'}:
    raise RuntimeError(f'Unexpected split labels: {sorted(split["split"].unique())}')
if set(split['userid']) != set(pendikom_ils['userid']):
    raise RuntimeError('Frozen split and WITH_ILS cohort do not contain the same users.')

train_ids = split.loc[split['split'].eq('train'), 'userid'].tolist()
test_ids = split.loc[split['split'].eq('test'), 'userid'].tolist()

ils_train = (
    pd.DataFrame({'userid': train_ids})
    .merge(pendikom_ils, on='userid', how='left', validate='one_to_one')
)
ils_test = (
    pd.DataFrame({'userid': test_ids})
    .merge(pendikom_ils, on='userid', how='left', validate='one_to_one')
)

if len(ils_train) != 322 or len(ils_test) != 139:
    raise RuntimeError(
        f'Expected 322/139 train/test rows, got {len(ils_train)}/{len(ils_test)}.'
    )

split_audit = pd.concat([
    ils_train[['userid','courseid'] + ILS_CODE_COLS + ['ils_ls_class']].assign(split='train'),
    ils_test[['userid','courseid'] + ILS_CODE_COLS + ['ils_ls_class']].assign(split='test'),
], ignore_index=True)
split_audit.to_csv(OUTPUT_DIR/'70_30_groundtruth_split_audit.csv', index=False)
ils_train[['userid']].to_csv(OUTPUT_DIR/'70_30_groundtruth_train_userids.csv', index=False)
ils_test[['userid']].to_csv(OUTPUT_DIR/'70_30_groundtruth_test_userids.csv', index=False)

print('PENDIKOM source rows:', len(pendikom_all))
print('WITH_ILS cohort     :', len(pendikom_ils))
print('Frozen train/test   :', len(ils_train), '/', len(ils_test))
print(split_audit.groupby('split').size().rename('n').reset_index())


def _fit_q75_reference(train_frame, seconds_col, opportunity_col=None):
    cols = ['courseid', seconds_col] + ([opportunity_col] if opportunity_col else [])
    t = train_frame[cols].copy()
    if opportunity_col is not None:
        t = t[t[opportunity_col].gt(0)]
    positive = t[t[seconds_col].gt(0)]

    by_course = positive.groupby('courseid')[seconds_col].quantile(0.75)
    global_q75 = float(positive[seconds_col].quantile(0.75)) if len(positive) else 1.0
    if not np.isfinite(global_q75) or global_q75 <= 0:
        global_q75 = 1.0
    return by_course, global_q75


def _ratio_pct(num, den):
    n = pd.to_numeric(num, errors='coerce').fillna(0).to_numpy(float)
    d = pd.to_numeric(den, errors='coerce').fillna(0).to_numpy(float)
    out = np.zeros(len(n), dtype=float)
    mask = d > 0
    out[mask] = 100.0 * n[mask] / d[mask]
    return out


stay_refs = {
    'content': _fit_q75_reference(
        ils_train, 'content_stay_seconds', 'content_available'
    ),
    'exercise': _fit_q75_reference(
        ils_train, 'exercise_stay_seconds', 'assign_available'
    ),
    'overview': _fit_q75_reference(
        ils_train, 'overview_stay_seconds', None
    ),
}


def build_pendikom_numeric(frame):
    b = frame
    meta = {
        'userid': b['userid'].astype(int),
        'courseid': b['courseid'].astype(int),
        'total_activity': pd.to_numeric(b['total_activity'], errors='raise').astype(int),
        'active_days': pd.to_numeric(b['active_days'], errors='raise').astype(int),
    }
    if 'course_fullname' in b.columns:
        meta['course_fullname'] = b['course_fullname'].astype(str)

    out = pd.DataFrame(meta)
    out['content_visit'] = _ratio_pct(
        b['content_visit_evidence_count'], b['content_available']
    )

    ref_by_course, ref_global = stay_refs['content']
    ref = b['courseid'].map(ref_by_course).fillna(ref_global).to_numpy(float)
    sec = pd.to_numeric(b['content_stay_seconds'], errors='coerce').fillna(0).to_numpy(float)
    out['content_stay'] = np.where(sec > 0, 100.0 * sec / ref, 0.0)

    out['selfass_visit'] = _ratio_pct(
        b['question_attempt_count'], b['question_available']
    )
    out['selfass_twice_wrong'] = _ratio_pct(
        b['twice_wrong_question_count'], b['repeated_question_count']
    )
    out['exercise_visit'] = _ratio_pct(
        b['submitted_assignment_count'], b['assign_available']
    )

    ref_by_course, ref_global = stay_refs['exercise']
    ref = b['courseid'].map(ref_by_course).fillna(ref_global).to_numpy(float)
    sec = pd.to_numeric(b['exercise_stay_seconds'], errors='coerce').fillna(0).to_numpy(float)
    out['exercise_stay'] = np.where(sec > 0, 100.0 * sec / ref, 0.0)

    for tag in ['detail','overview','facts','concepts','graphics','text','interpret','develop']:
        numerator = (
            'interpret_question_score_sum'
            if tag == 'interpret'
            else f'ques_{tag}_fraction_sum'
        )
        out[f'ques_{tag}'] = _ratio_pct(
            b[numerator], b[f'ques_{tag}_attempted_count']
        )

    out['quiz_revisions'] = _ratio_pct(
        b['revised_question_attempt_count'], b['question_attempt_count']
    )

    events = pd.to_numeric(
        b['quiz_results_event_count'], errors='coerce'
    ).fillna(0).to_numpy(float)
    seconds = pd.to_numeric(
        b['quiz_results_stay_seconds_sum'], errors='coerce'
    ).fillna(0).to_numpy(float)
    q = np.zeros(len(b), dtype=float)
    mask = events > 0
    q[mask] = seconds[mask] / events[mask]
    out['quiz_stay_results'] = q

    module_views = np.maximum(
        pd.to_numeric(b['module_view_count'], errors='coerce').fillna(0).to_numpy(float),
        1.0,
    )
    out['navigation_skip'] = (
        100.0 *
        pd.to_numeric(b['navigation_skip_count'], errors='coerce').fillna(0).to_numpy(float)
        / module_views
    )
    out['navigation_overview_visit'] = (
        100.0 *
        pd.to_numeric(b['overview_visit_count'], errors='coerce').fillna(0).to_numpy(float)
        / module_views
    )

    ref_by_course, ref_global = stay_refs['overview']
    ref = b['courseid'].map(ref_by_course).fillna(ref_global).to_numpy(float)
    sec = pd.to_numeric(b['overview_stay_seconds'], errors='coerce').fillna(0).to_numpy(float)
    out['navigation_overview_stay'] = np.where(sec > 0, 100.0 * sec / ref, 0.0)

    return out


def categorize_pendikom(numeric_frame):
    meta_cols = [
        c for c in ['userid','courseid','course_fullname','total_activity','active_days']
        if c in numeric_frame.columns
    ]
    out = numeric_frame[meta_cols].copy()
    for feature in PENDIKOM_FEATURES:
        low, high = PENDIKOM_THRESHOLDS[feature]
        v = pd.to_numeric(numeric_frame[feature], errors='coerce').fillna(0).to_numpy(float)
        out[feature] = np.where(
            v < low, 1, np.where(v < high, 2, 3)
        ).astype(np.int8)
    return out


gt_train_numeric = build_pendikom_numeric(ils_train)
gt_test_numeric = build_pendikom_numeric(ils_test)
gt_train = categorize_pendikom(gt_train_numeric)
gt_test = categorize_pendikom(gt_test_numeric)

constant_features = [
    f for f in PENDIKOM_FEATURES
    if gt_train[f].nunique(dropna=False) == 1
]
features = [f for f in PENDIKOM_FEATURES if f not in constant_features]

gt_train_numeric.to_csv(OUTPUT_DIR/'70_30_groundtruth_train_numeric_19.csv', index=False)
gt_test_numeric.to_csv(OUTPUT_DIR/'70_30_groundtruth_test_numeric_19.csv', index=False)
gt_train.to_csv(OUTPUT_DIR/'70_30_groundtruth_train_categorical_19.csv', index=False)
gt_test.to_csv(OUTPUT_DIR/'70_30_groundtruth_test_categorical_19.csv', index=False)

feature_audit = pd.DataFrame({
    'feature': PENDIKOM_FEATURES,
    'n_unique_train': [int(gt_train[f].nunique(dropna=False)) for f in PENDIKOM_FEATURES],
    'used': [f in features for f in PENDIKOM_FEATURES],
})
feature_audit.to_csv(OUTPUT_DIR/'70_30_groundtruth_feature_audit.csv', index=False)

reference_audit_rows = []
for key, seconds_col, stored_col in [
    ('content','content_stay_seconds','content_stay_reference_seconds'),
    ('exercise','exercise_stay_seconds','exercise_stay_reference_seconds'),
    ('overview','overview_stay_seconds','overview_stay_reference_seconds'),
]:
    if stored_col not in pendikom_ils.columns:
        continue
    by_course, _ = stay_refs[key]
    for courseid, value in by_course.items():
        stored = (
            pendikom_ils.loc[pendikom_ils['courseid'].eq(courseid), stored_col]
            .dropna().unique()
        )
        if len(stored):
            reference_audit_rows.append({
                'source': key,
                'courseid': int(courseid),
                'computed_train_q75': float(value),
                'compiled_reference': float(stored[0]),
                'difference': float(value - float(stored[0])),
            })

reference_audit = pd.DataFrame(reference_audit_rows)
reference_audit.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_frozen_preprocessing_state_audit.csv',
    index=False,
)

print('Core numeric/categorical indicators:', len(PENDIKOM_FEATURES))
print('Used CARE-KM predictors            :', len(features))
print('Constant train features            :', constant_features)
print(feature_audit)


def pendikom_semantic_rules():
    rows = []
    def add(feature, dimension, low, high):
        rows.append({
            'feature': feature,
            'dimension': dimension,
            'low_supports': low,
            'high_supports': high,
        })

    for f in ['selfass_visit','selfass_twice_wrong','exercise_visit','exercise_stay']:
        add(f, 'Active/Reflective', 'Reflective', 'Active')
    for f in ['content_visit','content_stay','quiz_stay_results']:
        add(f, 'Active/Reflective', 'Active', 'Reflective')

    for f in ['selfass_visit','exercise_visit','ques_detail','ques_facts','quiz_revisions','quiz_stay_results']:
        add(f, 'Sensing/Intuitive', 'Intuitive', 'Sensing')
    for f in ['content_visit','content_stay','ques_concepts','ques_develop']:
        add(f, 'Sensing/Intuitive', 'Sensing', 'Intuitive')

    add('ques_graphics', 'Visual/Verbal', 'Verbal', 'Visual')
    for f in ['content_visit','ques_text']:
        add(f, 'Visual/Verbal', 'Visual', 'Verbal')

    add('ques_detail', 'Sequential/Global', 'Global', 'Sequential')
    for f in ['ques_overview','ques_interpret','ques_develop',
              'navigation_skip','navigation_overview_visit','navigation_overview_stay']:
        add(f, 'Sequential/Global', 'Sequential', 'Global')

    return pd.DataFrame(rows)


semantic_rules = pendikom_semantic_rules()
semantic_rules = semantic_rules[
    semantic_rules['feature'].isin(features)
].copy()
semantic_rules.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_semantic_rules.csv', index=False
)

Xtr_gt = gt_train[features].astype(str).to_numpy(dtype=object)
Xte_gt = gt_test[features].astype(str).to_numpy(dtype=object)

ext_init, ext_fit = fit_care(Xtr_gt)
ext_modes = np.asarray(ext_fit['modes'], dtype=object)
ext_train_cluster = np.asarray(ext_fit['labels'], dtype=int)
ext_test_cluster, ext_test_dist = ck.assign_nearest(Xte_gt, ext_modes)
ext_test_cluster = np.asarray(ext_test_cluster, dtype=int)

ext_map = semantic_mapping(
    Xtr_gt, ext_train_cluster, ext_modes, features, semantic_rules
)
ext_map.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_frozen_cluster_to_ls_mapping.csv',
    index=False,
)
pd.DataFrame(ext_modes, columns=features).assign(
    cluster_id=np.arange(K)
).to_csv(
    OUTPUT_DIR/'70_30_groundtruth_frozen_modes.csv',
    index=False,
)

mp = ext_map.set_index('cluster_id')

test_eval = ils_test[
    ['userid','courseid'] + ILS_CODE_COLS + ['ils_ls_class']
].copy().reset_index(drop=True)
test_eval['reference_cluster_id'] = ext_test_cluster
test_eval['distance_to_frozen_mode'] = ext_test_dist
for c in DIM_KEYS + ['ls_class','mapping_status']:
    test_eval[f'carekm_{c}'] = [
        mp.loc[int(k), c] for k in ext_test_cluster
    ]
test_eval.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_test_carekm_vs_ils_full.csv',
    index=False,
)

BASELINE_METHODS = ['CARE-KM','Bai','Cao','K-Modes++','HR-2023','Huang','Random']
BASELINE_STOCHASTIC = {'K-Modes++','HR-2023','Huang','Random'}
BASELINE_SEED = 4161600

baseline_raw_rows = []
baseline_summary_rows = []

for method in BASELINE_METHODS:
    reps = 15 if method in BASELINE_STOCHASTIC else 1
    method_rows = []

    for rep in range(reps):
        seed = BASELINE_SEED + rep
        row, _, _ = ck.run_method(
            method,
            Xtr_gt,
            None,
            K,
            seed=seed,
            care_config=CARE_CONFIG,
            si_sample_size=None,
            si_seed=seed,
            max_iter=MAX_ITER,
        )
        row = dict(row)
        row['rep'] = rep
        method_rows.append(row)
        baseline_raw_rows.append(row)

    method_df = pd.DataFrame(method_rows)
    baseline_summary_rows.append({
        'Method': method,
        'Runs': reps,
        'Initial cost': float(method_df['initial_cost'].mean()),
        'Final cost': float(method_df['final_cost'].mean()),
        'DBI': float(method_df['dbi'].mean()),
        'SI': float(method_df['si'].mean()),
        'Iterations': float(method_df['iterations'].mean()),
        'DBI SD': float(method_df['dbi'].std(ddof=1)) if reps > 1 else np.nan,
        'SI SD': float(method_df['si'].std(ddof=1)) if reps > 1 else np.nan,
    })

baseline_raw = pd.DataFrame(baseline_raw_rows)
baseline_summary = pd.DataFrame(baseline_summary_rows)

baseline_raw.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_clustering_baseline_raw.csv',
    index=False,
)
baseline_summary.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_clustering_baseline.csv',
    index=False,
)

print('PENDIKOM questionnaire cohort — DBI/SI vs baselines')
print(
    baseline_summary[
        ['Method','Runs','Initial cost','Final cost','DBI','SI','Iterations']
    ].round(4)
)


def add_dimension_metrics(rows, method, dimension, truth, pred):
    truth = np.asarray(truth, dtype=object)
    pred = np.asarray(pred, dtype=object)
    labels = sorted(set(truth.tolist()))

    acc = accuracy_score(truth, pred)
    ba = balanced_accuracy_score(truth, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, labels=labels, average='macro', zero_division=0
    )

    rows.extend([
        {'Method':method,'Measure':'Acc','Dimension':dimension,'Value':float(acc)},
        {'Method':method,'Measure':'BA','Dimension':dimension,'Value':float(ba)},
        {'Method':method,'Measure':'F1','Dimension':dimension,'Value':float(f1)},
    ])


metric_rows = []
care_map_cols = [
    'carekm_active_reflective',
    'carekm_sensing_intuitive',
    'carekm_visual_verbal',
    'carekm_sequential_global',
]

for dim, gt_col, care_col in zip(DIM_NAMES, ILS_CODE_COLS, care_map_cols):
    truth = test_eval[gt_col].map(ils_code_to_full).to_numpy(dtype=object)
    pred = test_eval[care_col].map(pole_abbr_to_full).to_numpy(dtype=object)
    add_dimension_metrics(metric_rows, 'CARE-KM', dim, truth, pred)

train_label = pd.DataFrame({
    'userid': gt_train['userid'].astype(int),
    'cluster_id': ext_train_cluster,
})
train_label['ls_class'] = [
    mp.loc[int(k), 'ls_class'] for k in ext_train_cluster
]
resolved = train_label['ls_class'].ne('AMBIGUOUS').to_numpy()

if resolved.sum() == 0:
    raise RuntimeError('No resolved CARE-KM semantic labels in training data.')

X_train_cls = gt_train.loc[resolved, features].astype(int).reset_index(drop=True)
y_train_cls = train_label.loc[resolved, 'ls_class'].astype(str).reset_index(drop=True)
X_test_cls = gt_test[features].astype(int).reset_index(drop=True)

pred_table = test_eval[
    ['userid'] + ILS_CODE_COLS + ['ils_ls_class']
].copy().reset_index(drop=True)

for model_name in ['NB','DT','SVM']:
    model = build_external_classifiers()[model_name]

    if model_name == 'NB':
        model.fit(X_train_cls.to_numpy(), y_train_cls)
        pred_ls = model.predict(X_test_cls.to_numpy())
    else:
        model.fit(X_train_cls, y_train_cls)
        pred_ls = model.predict(X_test_cls)

    pred_table[f'{model_name}_predicted_ls_class'] = pred_ls

    for di, (dim, gt_col) in enumerate(zip(DIM_NAMES, ILS_CODE_COLS)):
        truth = pred_table[gt_col].map(ils_code_to_full).to_numpy(dtype=object)
        pred_dim = np.asarray(
            [LS_TO_POLES[str(ls)][di] for ls in pred_ls],
            dtype=object,
        )
        pred_table[f'{model_name}_{DIM_KEYS[di]}'] = pred_dim
        add_dimension_metrics(metric_rows, model_name, dim, truth, pred_dim)
        save_confusion(
            truth, pred_dim,
            OUTPUT_DIR/f'70_30_groundtruth_{model_name}_{DIM_KEYS[di]}_confusion_matrix.csv',
        )

metrics_long = pd.DataFrame(metric_rows)
metrics_long.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_classifier_dimension_metrics.csv',
    index=False,
)
pred_table.to_csv(
    OUTPUT_DIR/'70_30_groundtruth_classifier_predictions.csv',
    index=False,
)

table11_rows = []
for method in ['NB','DT','SVM','CARE-KM']:
    for measure in ['Acc','BA','F1']:
        subset = (
            metrics_long[
                metrics_long['Method'].eq(method) &
                metrics_long['Measure'].eq(measure)
            ]
            .set_index('Dimension')['Value']
        )
        row = {'Method and measure': f'{method} {measure}'}
        for dim in DIM_NAMES:
            row[dim] = float(subset[dim])
        row['Mean'] = float(subset.mean())
        table11_rows.append(row)

table11_numeric = pd.DataFrame(table11_rows)
table11_numeric.to_csv(
    OUTPUT_DIR/'70_30_dimensional_performance_vs_ils_numeric.csv',
    index=False,
)

table11_paper = table11_numeric.copy()
for c in DIM_NAMES + ['Mean']:
    table11_paper[c] = table11_paper[c].map(lambda x: f'{100*x:.2f}%')
table11_paper.to_csv(
    OUTPUT_DIR/'70_30_dimensional_performance_vs_ils.csv',
    index=False,
)

classifier_accuracy_rows = []
for method in ['NB','DT','SVM']:
    subset = (
        metrics_long[
            metrics_long['Method'].eq(method) &
            metrics_long['Measure'].eq('Acc')
        ]
        .set_index('Dimension')['Value']
    )
    classifier_accuracy_rows.append({
        'Classifier': method,
        **{dim: float(subset[dim]) for dim in DIM_NAMES},
        'Mean Dimensional Accuracy': float(subset.mean()),
    })

classifier_table = pd.DataFrame(classifier_accuracy_rows)
classifier_table.to_csv(
    OUTPUT_DIR/'70_30_table_classifier_dimensional_accuracy_vs_ils_numeric.csv',
    index=False,
)
classifier_paper = classifier_table.copy()
for c in DIM_NAMES + ['Mean Dimensional Accuracy']:
    classifier_paper[c] = classifier_paper[c].map(lambda x: f'{100*x:.2f}%')
classifier_paper.to_csv(
    OUTPUT_DIR/'70_30_table_classifier_dimensional_accuracy_vs_ils.csv',
    index=False,
)

care_acc = (
    metrics_long[
        metrics_long['Method'].eq('CARE-KM') &
        metrics_long['Measure'].eq('Acc')
    ]
    .set_index('Dimension')['Value']
)
external_rows = [
    {'FSLSM Dimension': dim, 'Agreement': float(care_acc[dim])}
    for dim in DIM_NAMES
]
external_rows.append({
    'FSLSM Dimension':'Mean dimensional agreement',
    'Agreement':float(care_acc.mean()),
})
external_numeric = pd.DataFrame(external_rows)
external_numeric.to_csv(
    OUTPUT_DIR/'70_30_table_external_validation_carekm_vs_ils_numeric.csv',
    index=False,
)
external_paper = external_numeric.copy()
external_paper['Agreement'] = external_paper['Agreement'].map(
    lambda x: f'{100*x:.2f}%'
)
external_paper.to_csv(
    OUTPUT_DIR/'70_30_table_external_validation_carekm_vs_ils.csv',
    index=False,
)

# Bentuk tabel ringkas yang mengikuti susunan Table 12 pada manuskrip.
table12_rows = []
for method, label in [('CARE-KM','Direct cluster-derived profile'),('NB','Naive Bayes'),('DT','Decision Tree'),('SVM','SVM')]:
    acc = metrics_long[(metrics_long['Method'].eq(method)) & (metrics_long['Measure'].eq('Acc'))].set_index('Dimension')['Value']
    ba = metrics_long[(metrics_long['Method'].eq(method)) & (metrics_long['Measure'].eq('BA'))].set_index('Dimension')['Value']
    f1 = metrics_long[(metrics_long['Method'].eq(method)) & (metrics_long['Measure'].eq('F1'))].set_index('Dimension')['Value']
    table12_rows.append({
        'Evidence / Model': label,
        'A/R Acc.': float(acc['Active / Reflective']),
        'S/I Acc.': float(acc['Sensing / Intuitive']),
        'V/V Acc.': float(acc['Visual / Verbal']),
        'S/G Acc.': float(acc['Sequential / Global']),
        'Mean Acc.': float(acc.mean()),
        'Mean BA': float(ba.mean()),
        'Mean F1': float(f1.mean()),
    })
table12_numeric = pd.DataFrame(table12_rows)
table12_numeric.to_csv(OUTPUT_DIR/'Table_12_External_ILS_Correspondence_numeric.csv', index=False)
table12_paper = table12_numeric.copy()
for c in ['A/R Acc.','S/I Acc.','V/V Acc.','S/G Acc.','Mean Acc.','Mean BA','Mean F1']:
    table12_paper[c] = table12_paper[c].map(lambda x: f'{100*x:.2f}%')
table12_paper.to_csv(OUTPUT_DIR/'Table_12_External_ILS_Correspondence.csv', index=False)

print('Resolved train labels:', int(resolved.sum()), '/', len(resolved))
print('Table 12. External correspondence against independent ILS labels')
print(table12_paper)


care_row = baseline_summary.loc[
    baseline_summary['Method'].eq('CARE-KM')
].iloc[0]

summary = {
    'source': 'DATA/PENDIKOM/compiled_pendikom_moodle_logs.csv',
    'repository': 'https://github.com/Dewi2501/Learning-Style-Based',
    'selection': 'has_ils == 1',
    'source_rows': int(len(pendikom_all)),
    'ils_rows': int(len(pendikom_ils)),
    'train_rows': int(len(ils_train)),
    'test_rows': int(len(ils_test)),
    'split_file': 'DATA/PENDIKOM/pendikom_ils_split_161600.csv',
    'split_seed_reference': SPLIT_SEED,
    'core_features': int(len(PENDIKOM_FEATURES)),
    'used_features': int(len(features)),
    'constant_features': constant_features,
    'resolved_train_learners': int(resolved.sum()),
    'unresolved_train_learners': int((~resolved).sum()),
    'care_train_dbi': float(care_row['DBI']),
    'care_train_si': float(care_row['SI']),
    'care_config': CARE_CONFIG,
    'parent_initializer': 'Cao',
    'refinement_budget_rule': 'ceil(log2(candidate_budget))',
    'derived_refinement_budget': DERIVED_REFINEMENT_BUDGET,
    'exchange_depth_is_independent_hyperparameter': False,
    'care_implementation_version': str(ck.CARE_IMPLEMENTATION_VERSION),
    'note': (
        'ILS labels are not used to form CARE-KM clusters or semantic mapping; '
        'they are used as the independent questionnaire reference on the frozen holdout.'
    ),
}
(OUTPUT_DIR/'70_30_pendikom_questionnaire_summary.json').write_text(
    json.dumps(summary, indent=2),
    encoding='utf-8',
)
print(json.dumps(summary, indent=2))
