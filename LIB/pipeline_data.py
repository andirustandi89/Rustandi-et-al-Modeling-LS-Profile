"""Pembaca dataset behavioral, pembagian holdout, dan label ILS independen."""
from pathlib import Path
import csv, hashlib, json, os
import numpy as np
import pandas as pd
from pipeline_paths import find_root, lms_data, pendikom_data, protocol_path

MAIN_FEATURES=['content_visit','content_stay','selfass_visit_proxy','selfass_stay_proxy',
 'exercise_visit_proxy','exercise_stay_proxy','forum_visit','forum_stay','forum_post',
 'quiz_stay_results_proxy','navigation_overview_visit','navigation_overview_stay','return_to_overview_rate']
PENDIKOM_FEATURES=['content_visit','content_stay','selfass_visit','selfass_twice_wrong',
 'exercise_visit','exercise_stay','ques_detail','ques_overview','ques_facts','ques_concepts',
 'ques_graphics','ques_text','ques_interpret','ques_develop','quiz_revisions','quiz_stay_results',
 'navigation_skip','navigation_overview_visit','navigation_overview_stay']
PENDIKOM_USED=[x for x in PENDIKOM_FEATURES if x!='navigation_skip']
ILS_CODE_COLS=['ils_active_reflective_code','ils_sensing_intuitive_code','ils_visual_verbal_code','ils_sequential_global_code']

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




def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def preflight(root=None, benchmarks=False):
    """Periksa keberadaan masukan dan skema yang diperlukan tahap eksperimen."""
    root=find_root(root)
    paths={'lms_numeric':lms_data('numeric',root),'lms_categorical':lms_data('categorical',root),
           'pendikom_numeric':pendikom_data('numeric',root),'pendikom_categorical':pendikom_data('categorical',root)}
    for key,path in paths.items():
        if not path.is_file():raise FileNotFoundError(f'Jalankan praproses terlebih dahulu: {path}')
        required=(['user']+MAIN_FEATURES) if key.startswith('lms_') else (['userid','courseid','split']+PENDIKOM_USED)
        columns=pd.read_csv(path,nrows=0).columns.tolist()
        if not set(required).issubset(columns):raise RuntimeError(f'Kolom masukan belum lengkap: {path}')
    for name in ('lms_split_indices.csv','pendikom_ils_labels.csv','pendikom_feature_plan.csv','dataset_metadata.json'):
        if not protocol_path(name,root).is_file():raise FileNotFoundError(protocol_path(name,root))
    if benchmarks:
        import generalization_data as gd
        if gd.resolve_dataset_keys()!=gd.DATASET_ORDER:raise RuntimeError('Eksperimen memakai sepuluh dataset benchmark dalam registry.')
        for key,value in [('CARE_KM_GEN_FULL_REPS','10'),('CARE_KM_GEN_RESAMPLE_REPS','30')]:
            if os.environ.get(key,value)!=value:raise RuntimeError(f'{key} harus {value} untuk protokol ini.')
        if float(os.environ.get('CARE_KM_GEN_FRACTION','0.80'))!=0.8:raise RuntimeError('Fraksi resampling harus 0.8.')
        for key in gd.resolve_dataset_keys():
            path=root/'DATA'/gd.DATASETS[key]['file']
            if not path.is_file():raise FileNotFoundError(path)
    return {key:str(path) for key,path in paths.items()}

def save(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,lineterminator='\n')

def prepare_splits(root=None):
    root=find_root(root);preflight(root)
    dest=root/'OUTPUT/SPLIT'
    main=pd.read_csv(lms_data('numeric',root),dtype={'user':str,'userid':str})
    main=main[['user']+MAIN_FEATURES].copy()
    for c in MAIN_FEATURES:main[c]=pd.to_numeric(main[c],errors='raise').astype(float)
    indices=pd.read_csv(protocol_path('lms_split_indices.csv',root),dtype={'user':str})
    assert indices['source_row_index'].is_unique and len(indices)==len(main)==6658
    if main.iloc[indices['source_row_index'].to_numpy()]['user'].tolist()!=indices['user'].tolist():
        raise RuntimeError('Urutan peserta LMS tidak sesuai keanggotaan holdout yang ditetapkan.')
    for label,n in [('train',4660),('test',1998)]:
        assert int(indices['split'].eq(label).sum())==n
    train=main.iloc[indices.loc[indices['split'].eq('train'),'source_row_index']].reset_index(drop=True)
    test=main.iloc[indices.loc[indices['split'].eq('test'),'source_row_index']].reset_index(drop=True)
    state,thresholds=fit_main_categorizer(train)
    for label,frame in [('TRAIN',train),('TEST',test)]:
        save(frame,dest/'LMS'/label/'numeric.csv')
        save(apply_main_categorizer(frame,state),dest/'LMS'/label/'categorical.csv')
    save(indices,dest/'LMS/70_30_main_lms_split_indices.csv')
    save(thresholds,dest/'LMS/70_30_main_lms_train_only_thresholds.csv')
    (dest/'LMS/train_categorizer_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')

    cohort=pd.read_csv(protocol_path('pendikom_ils_labels.csv',root)).drop(columns=['split'])
    if len(cohort)!=461 or cohort['userid'].duplicated().any():raise RuntimeError('Referensi ILS berubah.')
    numeric=pd.read_csv(pendikom_data('numeric',root))
    categorical=pd.read_csv(pendikom_data('categorical',root))
    actual_audit=[]
    for label,n in [('train',322),('test',139)]:
        nums=numeric.loc[numeric['split'].eq(label)].reset_index(drop=True)
        cats=categorical.loc[categorical['split'].eq(label)].reset_index(drop=True)
        assert len(nums)==len(cats)==n
        assert nums[['userid','courseid']].equals(cats[['userid','courseid']])
        ids=nums[['userid','courseid']]
        ils=ids.merge(cohort,on=['userid','courseid'],how='left',validate='one_to_one')
        if ils[ILS_CODE_COLS+['ils_ls_class']].isna().any().any():raise RuntimeError('Label ILS hilang.')
        actual_audit.append(ils[['userid','courseid']+ILS_CODE_COLS+['ils_ls_class']].assign(split=label))
        save(nums,dest/'PENDIKOM'/label.upper()/'numeric.csv')
        save(cats,dest/'PENDIKOM'/label.upper()/'categorical.csv')
        save(ils,dest/'PENDIKOM'/label.upper()/'reference_ils.csv')
        save(ids[['userid']],dest/'PENDIKOM'/f'70_30_groundtruth_{label}_userids.csv')
    combined=pd.concat(actual_audit,ignore_index=True)
    save(combined,dest/'PENDIKOM/70_30_groundtruth_split_audit.csv')
    mask=pd.read_csv(protocol_path('pendikom_feature_plan.csv',root))
    save(mask,dest/'PENDIKOM/70_30_groundtruth_feature_audit.csv')
    control=root/'OUTPUT/KONTROL';control.mkdir(parents=True,exist_ok=True)
    (control/'split_manifest.json').write_text(json.dumps({'input_files':{key:str(value) for key,value in preflight(root).items()},
        'LMS':{'train':4660,'test':1998},'PENDIKOM':{'train':322,'test':139},
        'split_source':'keanggotaan peserta dari protokol tetap','features_pendikom':PENDIKOM_USED},indent=2),encoding='utf-8')
    return dest

def load_lms_holdout(root=None):
    root=find_root(root);d=root/'OUTPUT/SPLIT/LMS'
    prepare_splits(root)
    main_num=pd.read_csv(lms_data('numeric',root),dtype={'user':str})
    main_num=main_num[['user']+MAIN_FEATURES].copy()
    for c in MAIN_FEATURES:main_num[c]=pd.to_numeric(main_num[c],errors='raise').astype(float)
    def read(part,name):return pd.read_csv(d/part/name,dtype={'user':str})
    train_raw=read('TRAIN','numeric.csv');test_raw=read('TEST','numeric.csv')
    train=read('TRAIN','categorical.csv');test=read('TEST','categorical.csv')
    for f in MAIN_FEATURES:train[f]=train[f].astype(int);test[f]=test[f].astype(int)
    return {'main_num':main_num,'main_train_raw':train_raw,'main_test_raw':test_raw,
            'main_train':train,'main_test':test,'MAIN_MODE':'numeric_train_only_discretization',
            'main_state':json.loads((d/'train_categorizer_state.json').read_text()),
            'main_thresholds':pd.read_csv(d/'70_30_main_lms_train_only_thresholds.csv')}

def load_pendikom_holdout(root=None):
    root=find_root(root);d=root/'OUTPUT/SPLIT/PENDIKOM'
    prepare_splits(root)
    cohort=pd.read_csv(protocol_path('pendikom_ils_labels.csv',root)).drop(columns=['split'])
    metadata=json.loads(protocol_path('dataset_metadata.json',root).read_text())
    def read(part,name):return pd.read_csv(d/part/name)
    return {'PENDIKOM_FEATURES':PENDIKOM_FEATURES,'features':PENDIKOM_USED,
        'constant_features':['navigation_skip'],'ILS_CODE_COLS':ILS_CODE_COLS,
        'pendikom_ils':cohort,'PENDIKOM_SOURCE_ROWS':int(metadata['pendikom_source_rows']),
        'ils_train':read('TRAIN','reference_ils.csv'),'ils_test':read('TEST','reference_ils.csv'),
        'gt_train_numeric':read('TRAIN','numeric.csv'),'gt_test_numeric':read('TEST','numeric.csv'),
        'gt_train':read('TRAIN','categorical.csv'),'gt_test':read('TEST','categorical.csv'),
        'feature_audit':pd.read_csv(d/'70_30_groundtruth_feature_audit.csv')}
