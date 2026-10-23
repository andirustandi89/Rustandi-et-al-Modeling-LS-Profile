# Menjalankan frozen 70:30 holdout pada cohort LMS utama.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
from io import BytesIO
import os, sys, json, shutil, zipfile, subprocess, hashlib, itertools, importlib.util, csv, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


PROJECT_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = PROJECT_ROOT / 'DATA'
PYTHON_DIR = PROJECT_ROOT / 'CODE'
LIB_DIR = PROJECT_ROOT / 'CODE' / 'LIB'
OUTPUT_DIR = PROJECT_ROOT / 'OUTPUT' / '08_MAIN_LMS_HOLDOUT'
DATA_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))
sys.modules.pop('care_km', None)
import care_km as ck

SPLIT_SEED = 161600
TEST_SIZE = 0.30
K = 16
MAX_ITER = 100
CARE_CONFIG = dict(candidate_rule='separation', candidate_budget=16, min_gain=0.0)
DERIVED_REFINEMENT_BUDGET = ck.derive_refinement_budget(CARE_CONFIG['candidate_budget'])
assert DERIVED_REFINEMENT_BUDGET == 4
PIPELINE_VERSION = 'carekm-holdout-v6'

MAIN_FEATURES = [
    'content_visit','content_stay','selfass_visit_proxy','selfass_stay_proxy',
    'exercise_visit_proxy','exercise_stay_proxy','forum_visit','forum_stay','forum_post',
    'quiz_stay_results_proxy','navigation_overview_visit','navigation_overview_stay','return_to_overview_rate'
]

DIM_NAMES = ['Active / Reflective','Sensing / Intuitive','Visual / Verbal','Sequential / Global']
DIM_KEYS = ['active_reflective','sensing_intuitive','visual_verbal','sequential_global']

print('PROJECT_ROOT   :', PROJECT_ROOT)
print('OUTPUT_DIR     :', OUTPUT_DIR)
print('Versi pipeline holdout utama:', PIPELINE_VERSION)
print('CARE implementation:', ck.CARE_IMPLEMENTATION_VERSION)
print('Derived refinement budget:', DERIVED_REFINEMENT_BUDGET)


assert ck.CARE_IMPLEMENTATION_VERSION == 'encoded-incremental-derived-budget-v3'
assert ck.derive_refinement_budget(CARE_CONFIG['candidate_budget']) == DERIVED_REFINEMENT_BUDGET
print('Pemeriksaan source CARE-KM: LULUS')


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
LS_LOOKUP = {tuple(combo): f'LS{i:02d}' for i, combo in enumerate(itertools.product(*[POLES[d] for d in DIMENSION_ORDER]), start=1)}
LS_TO_POLES = {v:k for k,v in LS_LOOKUP.items()}


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


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

def build_main_classifiers():

    return {
        'NB': categorical_nb(),
        'DT': Pipeline([('onehot', one_hot_encoder()), ('model', DecisionTreeClassifier(random_state=SPLIT_SEED))]),
        'SVM': Pipeline([('onehot', one_hot_encoder()), ('model', SVC(kernel='rbf', C=1.0, gamma='scale'))]),
    }

def build_external_classifiers():

    return {
        'NB': categorical_nb(),
        'DT': DecisionTreeClassifier(random_state=SPLIT_SEED, class_weight='balanced'),
        'SVM': Pipeline([('onehot', one_hot_encoder()), ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'))]),
    }


def fit_care(X):
    X=np.asarray(X,dtype=object)
    init=ck.care_init(X,K,**CARE_CONFIG)
    result=ck.kmodes_fit(X,init,max_iter=MAX_ITER)
    return init,result


def signed_vote(value, rule, features):
    value=int(value)
    first,second=POLES[str(rule.dimension)]
    if value==2:
        return 0
    if hasattr(rule,'high_occurrence_indicates'):
        pole = str(rule.high_occurrence_indicates) if value==3 else str(rule.low_occurrence_indicates)
    else:
        pole = str(rule.high_supports) if value==3 else str(rule.low_supports)
    if pole==first: return 1
    if pole==second: return -1
    raise ValueError(f'Unknown semantic pole: {pole}')


def semantic_mapping(X, labels, modes, features, semantic_rules):
    X=np.asarray(X,dtype=object); labels=np.asarray(labels,dtype=int); modes=np.asarray(modes,dtype=object)
    fpos={f:i for i,f in enumerate(features)}
    rows=[]
    for cid in range(K):
        selected=[]; short=[]
        for dim in DIMENSION_ORDER:
            rules=semantic_rules[semantic_rules['dimension'].astype(str).eq(dim)]
            mv=np.asarray([signed_vote(modes[cid,fpos[str(r.feature)]], r, features) for r in rules.itertuples(index=False)],dtype=int)
            ms=int(np.sum(mv>0)-np.sum(mv<0))
            member_X=X[labels==cid]
            nv=[]
            for r in rules.itertuples(index=False):
                j=fpos[str(r.feature)]
                nv.extend(signed_vote(v,r,features) for v in member_X[:,j])
            nv=np.asarray(nv,dtype=int)
            ns=int(np.sum(nv>0)-np.sum(nv<0)) if len(nv) else 0
            first,second=POLES[dim]
            pole = first if ms>0 else second if ms<0 else first if ns>0 else second if ns<0 else None
            selected.append(pole)
            short.append('Balanced' if pole is None else ABBR[pole])
        unresolved=any(x is None for x in selected)
        rows.append({
            'cluster_id':cid,'cluster_name':f'C{cid+1:02d}',
            'cluster_size':int(np.sum(labels==cid)),
            'active_reflective':short[0], 'sensing_intuitive':short[1],
            'visual_verbal':short[2], 'sequential_global':short[3],
            'ls_class':'AMBIGUOUS' if unresolved else LS_LOOKUP[tuple(selected)],
            'mapping_status':'BALANCED' if unresolved else 'RESOLVED',
        })
    return pd.DataFrame(rows)


def main_semantic_rules():
    rows=[]
    def add(f,d,lo,hi): rows.append({'feature':f,'dimension':d,'low_supports':lo,'high_supports':hi})
    add('content_visit','Active/Reflective','Active','Reflective')
    add('content_stay','Active/Reflective','Active','Reflective')
    add('selfass_visit_proxy','Active/Reflective','Reflective','Active')
    add('selfass_stay_proxy','Active/Reflective','Active','Reflective')
    add('exercise_visit_proxy','Active/Reflective','Reflective','Active')
    add('exercise_stay_proxy','Active/Reflective','Reflective','Active')
    add('forum_visit','Active/Reflective','Active','Reflective')
    add('forum_post','Active/Reflective','Reflective','Active')
    add('quiz_stay_results_proxy','Active/Reflective','Active','Reflective')
    add('content_visit','Sensing/Intuitive','Sensing','Intuitive')
    add('content_stay','Sensing/Intuitive','Sensing','Intuitive')
    add('selfass_visit_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('selfass_stay_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('exercise_visit_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('quiz_stay_results_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('content_visit','Visual/Verbal','Visual','Verbal')
    add('forum_visit','Visual/Verbal','Visual','Verbal')
    add('forum_stay','Visual/Verbal','Visual','Verbal')
    add('forum_post','Visual/Verbal','Visual','Verbal')
    add('navigation_overview_visit','Sequential/Global','Sequential','Global')
    add('navigation_overview_stay','Sequential/Global','Sequential','Global')
    add('return_to_overview_rate','Sequential/Global','Sequential','Global')
    return pd.DataFrame(rows)


def mapping_to_targets(mapping, cluster_ids):
    mp=mapping.set_index('cluster_id')
    target=pd.DataFrame(index=np.arange(len(cluster_ids)))
    rename={
        'active_reflective':'Active / Reflective',
        'sensing_intuitive':'Sensing / Intuitive',
        'visual_verbal':'Visual / Verbal',
        'sequential_global':'Sequential / Global',
    }
    for col,new in rename.items():
        target[new]=[mp.loc[int(c),col] for c in cluster_ids]
    return target


def pole_abbr_to_full(x):
    m={'Act.':'Active','Ref.':'Reflective','Sens.':'Sensing','Int.':'Intuitive','Vis.':'Visual','Ver.':'Verbal','Seq.':'Sequential','Gl.':'Global'}
    return m.get(str(x), 'Balanced')


def ils_code_to_full(code):
    s=str(code)
    for p,full in [('Act.','Active'),('Ref.','Reflective'),('Sens.','Sensing'),('Int.','Intuitive'),('Vis.','Visual'),('Ver.','Verbal'),('Seq.','Sequential'),('Gl.','Global')]:
        if s.startswith(p): return full
    raise ValueError(f'Unexpected ILS code: {code}')


def evaluate_per_dimension(y_true, y_pred):
    return float(np.mean(np.asarray(y_true,dtype=str)==np.asarray(y_pred,dtype=str)))


def save_confusion(y_true, y_pred, path):
    labs=sorted(set(map(str,y_true))|set(map(str,y_pred)))
    pd.DataFrame(confusion_matrix(y_true,y_pred,labels=labs),index=labs,columns=labs).to_csv(path)

print('Helpers ready.')


main_numeric_candidates = [
    DATA_DIR/'MAIN_LMS'/'main_lms_numeric.csv',
    DATA_DIR/'MAIN_LMS'/'main_lms_numeric.csv',
    DATA_DIR/'MAIN_LMS'/'main_lms_numeric.csv',
    DATA_DIR/'MAIN_LMS'/'main_lms_numeric.csv',
]
MAIN_NUMERIC = next((p for p in main_numeric_candidates if p.is_file()), None)
MAIN_CATEGORICAL = DATA_DIR/'MAIN_LMS'/'main_lms_categorical.csv'

if MAIN_NUMERIC is not None:
    main_num=pd.read_csv(MAIN_NUMERIC,dtype={'user':str,'userid':str})
    if 'user' not in main_num.columns and 'userid' in main_num.columns:
        main_num=main_num.rename(columns={'userid':'user'})
    missing=[c for c in ['user']+MAIN_FEATURES if c not in main_num.columns]
    if missing: raise RuntimeError(f'Main numeric source missing columns: {missing}')
    main_num=main_num[['user']+MAIN_FEATURES].copy()
    for c in MAIN_FEATURES: main_num[c]=pd.to_numeric(main_num[c],errors='raise').astype(float)
    MAIN_MODE='numeric_train_only_discretization'
    print('Main source:', MAIN_NUMERIC)
elif MAIN_CATEGORICAL.is_file():
    main_num=pd.read_csv(MAIN_CATEGORICAL,dtype={'user':str})[['user']+MAIN_FEATURES].copy()
    for c in MAIN_FEATURES: main_num[c]=pd.to_numeric(main_num[c],errors='raise').astype(int)
    MAIN_MODE='already_categorical_fallback'
    print('WARNING: numeric main source not found; using already-categorical main matrix.')
else:
    raise FileNotFoundError('Main LMS numeric/categorical source not found. Expected OUTPUT/01_PREPROCESSING/2dataset_numerc.csv or DATA/dataset_categorical_skala.csv')

if main_num['user'].duplicated().any(): raise RuntimeError('Duplicate main LMS user IDs.')
idx=np.arange(len(main_num))
tr_idx,te_idx=train_test_split(idx,test_size=TEST_SIZE,random_state=SPLIT_SEED,shuffle=True)
main_train_raw=main_num.iloc[tr_idx].reset_index(drop=True)
main_test_raw=main_num.iloc[te_idx].reset_index(drop=True)

pd.DataFrame({
    'source_row_index':np.r_[tr_idx,te_idx],
    'user':np.r_[main_num.iloc[tr_idx]['user'].to_numpy(),main_num.iloc[te_idx]['user'].to_numpy()],
    'split':['train']*len(tr_idx)+['test']*len(te_idx)
}).to_csv(OUTPUT_DIR/'70_30_main_lms_split_indices.csv',index=False)

print('Main rows:',len(main_num),'| train:',len(main_train_raw),'| untouched test:',len(main_test_raw))


DEGENERATE_MAX_SHARE=0.95
FEATURE_PLAN = [
    {'feature':'content_visit','mode':'GRAF_FIXED','low':75.0,'high':100.0},
    {'feature':'content_stay','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'selfass_visit_proxy','mode':'GRAF_FIXED','low':25.0,'high':75.0},
    {'feature':'selfass_stay_proxy','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'exercise_visit_proxy','mode':'GRAF_FIXED','low':25.0,'high':75.0},
    {'feature':'exercise_stay_proxy','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'forum_visit','mode':'GRAF_DEGENERACY','low':7.0,'high':50.0},
    {'feature':'forum_stay','mode':'GRAF_DEGENERACY','low':5.0,'high':30.0},
    {'feature':'forum_post','mode':'GRAF_SPARSE','low':1.0,'high':10.0},
    {'feature':'quiz_stay_results_proxy','mode':'GRAF_FIXED','low':30.0,'high':60.0},
    {'feature':'navigation_overview_visit','mode':'GRAF_DEGENERACY','low':10.0,'high':20.0},
    {'feature':'navigation_overview_stay','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'return_to_overview_rate','mode':'EMPIRICAL','low':np.nan,'high':np.nan},
]

def classify_cut(x,lo,hi):
    x=pd.to_numeric(x,errors='coerce').fillna(0).astype(float)
    return pd.Series(np.where(x<lo,1,np.where(x<hi,2,3)),index=x.index,dtype='int8')

def positive_q(x):
    x=pd.to_numeric(x,errors='coerce').fillna(0).astype(float); pos=x[x>0]
    if pos.empty: raise RuntimeError('Positive quantile unavailable.')
    lo=float(pos.quantile(1/3)); hi=float(pos.quantile(2/3))
    if hi<=lo:
        u=np.sort(pos.unique()); lo=float(np.quantile(u,1/3)); hi=float(np.quantile(u,2/3))
    if hi<=lo: raise RuntimeError('Degenerate train threshold.')
    return lo,hi

def share123(c): return c.value_counts(normalize=True).reindex([1,2,3],fill_value=0.0)

def fit_main_categorizer(train):
    state={}; audit=[]
    for sp in FEATURE_PLAN:
        f=sp['feature']; mode=sp['mode']; x=train[f]; lo=sp['low']; hi=sp['high']; method=mode
        if mode=='EMPIRICAL': lo,hi=positive_q(x)
        elif mode=='GRAF_DEGENERACY':
            c0=classify_cut(x,lo,hi); sh=share123(c0)
            if sh.max()>DEGENERATE_MAX_SHARE or (sh==0).any(): lo,hi=positive_q(x); method='GRAF_FALLBACK_POSITIVE_Q33_Q67'
        elif mode=='GRAF_SPARSE':
            c0=classify_cut(x,lo,hi); sh=share123(c0)
            if sh.max()>DEGENERATE_MAX_SHARE or (sh==0).any():
                pos=pd.to_numeric(x,errors='coerce').fillna(0); pos=pos[pos>0]
                lo=0.0; hi=float(pos.median()) if len(pos) else 1.0; method='ZERO_AWARE_POSITIVE_MEDIAN'
        state[f]={'low':float(lo),'high':float(hi),'method':method}
        audit.append({'feature':f,**state[f]})
    return state,pd.DataFrame(audit)

def apply_main_categorizer(df,state):
    out=pd.DataFrame({'user':df['user'].astype(str)})
    for f in MAIN_FEATURES:
        st=state[f]; x=pd.to_numeric(df[f],errors='coerce').fillna(0).astype(float)
        if st['method']=='ZERO_AWARE_POSITIVE_MEDIAN':
            out[f]=np.where(x<=0,1,np.where(x<st['high'],2,3)).astype(int)
        else:
            out[f]=classify_cut(x,st['low'],st['high']).astype(int)
    return out

if MAIN_MODE=='numeric_train_only_discretization':
    main_state,main_thresholds=fit_main_categorizer(main_train_raw)
    main_train=apply_main_categorizer(main_train_raw,main_state)
    main_test=apply_main_categorizer(main_test_raw,main_state)
    main_thresholds.to_csv(OUTPUT_DIR/'70_30_main_lms_train_only_thresholds.csv',index=False)
else:
    main_train=main_train_raw.copy(); main_test=main_test_raw.copy(); main_state={}

print('Main preprocessing mode:',MAIN_MODE)


Xtr=main_train[MAIN_FEATURES].astype(str).to_numpy(dtype=object)
Xte=main_test[MAIN_FEATURES].astype(str).to_numpy(dtype=object)
main_init,main_fit=fit_care(Xtr)
main_modes=main_fit['modes']; main_train_cluster=main_fit['labels']
main_test_cluster,main_test_dist=ck.assign_nearest(Xte,main_modes)
main_rules=main_semantic_rules()
main_map=semantic_mapping(Xtr,main_train_cluster,main_modes,MAIN_FEATURES,main_rules)
main_map.to_csv(OUTPUT_DIR/'70_30_main_lms_frozen_cluster_to_fslsm_mapping.csv',index=False)
pd.DataFrame(main_modes,columns=MAIN_FEATURES).assign(cluster_id=np.arange(K)).to_csv(OUTPUT_DIR/'70_30_main_lms_frozen_modes.csv',index=False)

main_y_train=mapping_to_targets(main_map,main_train_cluster).map(pole_abbr_to_full)
main_y_test=mapping_to_targets(main_map,main_test_cluster).map(pole_abbr_to_full)

rows=[]; metric_rows=[]; pred_out=main_test[['user']].copy(); pred_out['reference_cluster_id']=main_test_cluster; pred_out['distance_to_frozen_mode']=main_test_dist
for model_name,model in build_main_classifiers().items():
    accs=[]; bas=[]; f1s=[]
    for dim in DIM_NAMES:
        ytr=main_y_train[dim].astype(str); yte=main_y_test[dim].astype(str)
        if model_name=='NB':
            mdl=build_main_classifiers()['NB']; mdl.fit(main_train[MAIN_FEATURES].astype(int).to_numpy(),ytr); pred=mdl.predict(main_test[MAIN_FEATURES].astype(int).to_numpy())
        else:
            mdl=build_main_classifiers()[model_name]; mdl.fit(main_train[MAIN_FEATURES].astype(int),ytr); pred=mdl.predict(main_test[MAIN_FEATURES].astype(int))
        acc=accuracy_score(yte,pred)
        ba=balanced_accuracy_score(yte,pred)
        f1=precision_recall_fscore_support(yte,pred,average='macro',zero_division=0)[2]
        accs.append(acc); bas.append(ba); f1s.append(f1)
        metric_rows.append({'Classifier':model_name,'Dimension':dim,'Accuracy':acc,'Balanced Accuracy':ba,'Macro-F1':f1})
        pred_out[f'{model_name}_{dim}']=pred
    rows.append({
        'Classifier':model_name,
        **{DIM_NAMES[i]:accs[i] for i in range(4)},
        'Mean Dimensional Accuracy':float(np.mean(accs)),
        'Mean Balanced Accuracy':float(np.mean(bas)),
        'Mean Macro-F1':float(np.mean(f1s)),
    })

main_table=pd.DataFrame(rows).set_index('Classifier').loc[['NB','DT','SVM']].reset_index()
main_table.to_csv(OUTPUT_DIR/'70_30_main_lms_classifier_performance_numeric.csv',index=False)
pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR/'70_30_main_lms_classifier_dimension_metrics.csv',index=False)

accuracy_cols=DIM_NAMES+['Mean Dimensional Accuracy']
main_accuracy=main_table[['Classifier']+accuracy_cols].copy()
main_accuracy.to_csv(OUTPUT_DIR/'70_30_main_lms_prospective_dimensional_accuracy_numeric.csv',index=False)
main_paper=main_accuracy.copy()
for c in accuracy_cols: main_paper[c]=main_paper[c].map(lambda x:f'{100*x:.2f}%')
main_paper.to_csv(OUTPUT_DIR/'70_30_main_lms_prospective_dimensional_accuracy.csv',index=False)

summary_rows=[
    ('Active / Reflective', 'Active / Reflective'),
    ('Sensing / Intuitive', 'Sensing / Intuitive'),
    ('Visual / Verbal', 'Visual / Verbal'),
    ('Sequential / Global', 'Sequential / Global'),
    ('Mean accuracy', 'Mean Dimensional Accuracy'),
    ('Mean balanced accuracy', 'Mean Balanced Accuracy'),
    ('Mean macro-F1', 'Mean Macro-F1'),
]
paper_rows=[]
for label,col in summary_rows:
    row={'Dimension or summary':label}
    for model_name in ['NB','DT','SVM']:
        value=float(main_table.loc[main_table['Classifier'].eq(model_name),col].iloc[0])
        row[model_name]=value
    paper_rows.append(row)
paper_numeric=pd.DataFrame(paper_rows)
paper_numeric.to_csv(OUTPUT_DIR/'Table_11_Main_LMS_Classifier_Performance_numeric.csv',index=False)
paper_formatted=paper_numeric.copy()
for c in ['NB','DT','SVM']: paper_formatted[c]=paper_formatted[c].map(lambda x:f'{100*x:.2f}%')
paper_formatted.to_csv(OUTPUT_DIR/'Table_11_Main_LMS_Classifier_Performance.csv',index=False)

pred_out.to_csv(OUTPUT_DIR/'70_30_main_lms_prospective_predictions.csv',index=False)
print('Main LMS — classifier performance')
print(paper_formatted)
