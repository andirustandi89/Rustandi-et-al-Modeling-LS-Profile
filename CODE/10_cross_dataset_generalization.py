# Menjalankan evaluasi CARE-KM pada benchmark kategorikal.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

PIPELINE_VERSION = 'carekm-cross-dataset-v5'
print('Versi pipeline cross-dataset:', PIPELINE_VERSION)

from pathlib import Path
import json
import os
import sys
import hashlib

import numpy as np
import pandas as pd


PAPER_ROOT = Path(os.environ.get('CARE_KM_PAPER_ROOT', Path(__file__).resolve().parents[1])).resolve()

LIB_DIR = PAPER_ROOT / 'CODE' / 'LIB'
DATA_DIR = PAPER_ROOT / 'DATA' / 'BENCHMARKS'
OUT_DIR = PAPER_ROOT / 'OUTPUT' / '05_CROSS_DATASET'
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LIB_DIR))

for _name in ['experiment_lib', 'care_km', 'generalization_data']:
    sys.modules.pop(_name, None)
import experiment_lib as el
import care_km as ck
import generalization_data as gd


METHODS = list(el.METHODS)
CARE_CONFIG = dict(el.FROZEN_CARE_CONFIG)

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

print('PAPER_ROOT:', PAPER_ROOT)
print('DATA_DIR  :', DATA_DIR)
print('METHODS   :', METHODS)
print('CARE config:', CARE_CONFIG)


FULL_STOCHASTIC_REPS = int(os.environ.get('CARE_KM_GEN_FULL_REPS', '10'))
RESAMPLE_REPS = int(os.environ.get('CARE_KM_GEN_RESAMPLE_REPS', '30'))
FRACTION = float(os.environ.get('CARE_KM_GEN_FRACTION', '0.80'))
BASE_SEED = 120000
SI_SAMPLE_SIZE = el.SI_SAMPLE_SIZE
MAX_ITER = el.MAX_ITER

DATASETS = gd.resolve_dataset_keys()
CHECKPOINT_DIR = OUT_DIR / '10_GENERALIZATION_CHECKPOINTS'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

if not (0 < FRACTION <= 1):
    raise ValueError('FRACTION harus pada (0,1].')
if FULL_STOCHASTIC_REPS < 1 or RESAMPLE_REPS < 1:
    raise ValueError('Jumlah repetition harus >= 1.')

print('Datasets:', DATASETS)
print('Full stochastic reps:', FULL_STOCHASTIC_REPS)
print('Resample reps:', RESAMPLE_REPS)
print('Fraction:', FRACTION)


CORE_NUMERIC = [
    'initial_cost','final_cost','dbi','si','iterations',
    'min_cluster','max_cluster','cluster_cv',
    'init_sec','cluster_sec','total_sec',
]

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _load_checkpoint(path, dataset, id_col):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {'dataset','method',id_col, *CORE_NUMERIC}
    if not required.issubset(df.columns):
        raise RuntimeError(f'Checkpoint schema tidak sinkron: {path}')
    df = df[df['dataset'].astype(str) == dataset].copy()
    df = df[df['method'].astype(str).isin(METHODS)].copy()
    numeric = df[CORE_NUMERIC].apply(pd.to_numeric, errors='coerce')
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    df = df.loc[finite].copy()
    df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
    df = df[df[id_col].notna()].copy()
    df[id_col] = df[id_col].astype(int)
    return df.drop_duplicates(['method', id_col], keep='last').reset_index(drop=True)

def _checkpoint_meta(dataset, source_path, k):
    return {
        'dataset': dataset,
        'source_file': source_path.name,
        'source_sha256': sha256_file(source_path),
        'k': int(k),
        'methods': METHODS,
        'care_config': CARE_CONFIG,
        'care_metadata': el.frozen_care_metadata(),
        'care_implementation_version': str(ck.CARE_IMPLEMENTATION_VERSION),
        'care_km_sha256': _sha256(LIB_DIR / 'care_km.py'),
        'full_stochastic_reps': int(FULL_STOCHASTIC_REPS),
        'resample_reps': int(RESAMPLE_REPS),
        'fraction': float(FRACTION),
        'base_seed': int(BASE_SEED),
        'si_sample_size': int(SI_SAMPLE_SIZE),
        'max_iter': int(MAX_ITER),
    }

def _canonicalize_old_meta(old):
    """
    Menyamakan metadata checkpoint lama dengan skema yang dipakai saat ini.

    Konversi hanya menjembatani representasi q=16/E=4 ke derived budget; perubahan
    data, K, seed, pengulangan, metode, sampling SI, dan batas iterasi tetap diaudit.
    """
    x = dict(old)
    cfg = dict(x.get('care_config', {}))
    legacy_e = cfg.pop('max_exchanges', None)
    q = int(cfg.get('candidate_budget', 0) or 0)

    md = x.get('care_metadata')
    if md is None:

        derived_e = x.get('derived_refinement_budget', legacy_e)
        if derived_e is None and q >= 1:
            derived_e = int(ck.derive_refinement_budget(q))
        md = {
            **cfg,
            'parent_initializer': x.get('parent_initializer', 'Cao'),
            'refinement_budget_rule': x.get(
                'refinement_budget_rule', 'ceil(log2(candidate_budget))'
            ),
            'derived_refinement_budget': int(derived_e) if derived_e is not None else None,
            'exchange_depth_is_independent_hyperparameter': bool(
                x.get('exchange_depth_is_independent_hyperparameter', False)
            ),
        }
    else:
        md = dict(md)


    if legacy_e is not None:
        if q < 1 or int(legacy_e) != int(ck.derive_refinement_budget(q)):
            return None
        md['derived_refinement_budget'] = int(legacy_e)
        md['refinement_budget_rule'] = 'ceil(log2(candidate_budget))'
        md['exchange_depth_is_independent_hyperparameter'] = False
        md['parent_initializer'] = 'Cao'


    for key in [
        'parent_initializer','refinement_budget_rule',
        'derived_refinement_budget','exchange_depth_is_independent_hyperparameter'
    ]:
        x.pop(key, None)
    x['care_config'] = cfg
    x['care_metadata'] = md
    return x

def _meta_equivalent_except_source_stamp(old, new):
    """
    Mengembalikan True hanya bila protokol komputasinya setara.

    Versi implementasi dan hash CARE-KM diabaikan di fungsi ini karena perbedaannya
    memang dipakai untuk menentukan apakah baris CARE-KM perlu dihitung ulang.
    """
    old = dict(old)
    new = dict(new)
    old.pop('care_implementation_version', None)
    old.pop('care_km_sha256', None)
    new.pop('care_implementation_version', None)
    new.pop('care_km_sha256', None)
    return old == new

def _validate_or_write_meta(path, meta):
    """
    Memvalidasi protokol dan menentukan apakah baris CARE-KM perlu dihitung ulang.

    Metadata yang identik memakai checkpoint penuh. Metadata lama yang setara
    dimigrasikan dengan menghitung ulang CARE-KM saja. Perbedaan protokol yang
    substantif dihentikan dengan error agar hasil lama tidak tercampur.
    """
    if not path.exists():
        path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return False

    raw_old = json.loads(path.read_text(encoding='utf-8'))
    if raw_old == meta:
        return False

    old = _canonicalize_old_meta(raw_old)
    if old is None or not _meta_equivalent_except_source_stamp(old, meta):
        raise RuntimeError(
            f'Checkpoint protocol berbeda secara substantif untuk {meta["dataset"]}. '
            'Checkpoint tidak dimigrasikan otomatis karena data/K/seed/repetition/'
            'metode atau parameter evaluasi dapat berbeda.'
        )


    path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return True

def _full_summary(raw, dataset):
    agg = (
        raw.groupby('method')
        .agg(
            n=('run','count'),
            initial_cost=('initial_cost','mean'),
            final_cost=('final_cost','mean'),
            dbi=('dbi','mean'),
            si=('si','mean'),
            iterations=('iterations','mean'),
            min_cluster=('min_cluster','mean'),
            max_cluster=('max_cluster','mean'),
            cluster_cv=('cluster_cv','mean'),
            init_sec=('init_sec','mean'),
            cluster_sec=('cluster_sec','mean'),
            total_sec=('total_sec','mean'),
            ari=('ari','mean'),
            nmi=('nmi','mean'),
            ami=('ami','mean'),
        ).reset_index()
    )
    agg.insert(0, 'dataset', dataset)
    return agg

def _resample_summary(raw, dataset):
    agg = (
        raw.groupby('method')
        .agg(
            n=('rep','count'),
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
            min_cluster_mean=('min_cluster','mean'),
            min_cluster_sd=('min_cluster','std'),
            max_cluster_mean=('max_cluster','mean'),
            max_cluster_sd=('max_cluster','std'),
            cluster_cv_mean=('cluster_cv','mean'),
            cluster_cv_sd=('cluster_cv','std'),
            init_sec_mean=('init_sec','mean'),
            init_sec_sd=('init_sec','std'),
            cluster_sec_mean=('cluster_sec','mean'),
            cluster_sec_sd=('cluster_sec','std'),
            total_sec_mean=('total_sec','mean'),
            total_sec_sd=('total_sec','std'),
            ari_mean=('ari','mean'),
            nmi_mean=('nmi','mean'),
            ami_mean=('ami','mean'),
        ).reset_index()
    )
    agg.insert(0, 'dataset', dataset)
    return agg


all_full_raw = []
all_full_summary = []
all_resample_raw = []
all_resample_summary = []

for dataset in DATASETS:
    spec = gd.DATASETS[dataset]
    source_path = DATA_DIR / spec['file']
    X_df, y_s = gd.load_dataset(DATA_DIR, dataset)
    X_all = X_df.to_numpy(dtype=object)
    y_all = y_s.to_numpy(dtype=object)
    k = int(pd.Series(y_all).nunique())

    print('\n' + '='*90)
    print(f'{dataset}: n={len(X_all):,}, d={X_all.shape[1]}, K={k}')
    print('='*90)

    meta_path = CHECKPOINT_DIR / f'{dataset}_protocol.json'
    full_path = CHECKPOINT_DIR / f'{dataset}_full_raw.csv'
    res_path = CHECKPOINT_DIR / f'{dataset}_resample_raw.csv'
    recompute_care = _validate_or_write_meta(
        meta_path, _checkpoint_meta(dataset, source_path, k)
    )

    full_raw = _load_checkpoint(full_path, dataset, 'run')
    res_raw = _load_checkpoint(res_path, dataset, 'rep')

    if recompute_care:
        old_full_n = int((full_raw['method'].astype(str) == 'CARE-KM').sum()) if len(full_raw) else 0
        old_res_n = int((res_raw['method'].astype(str) == 'CARE-KM').sum()) if len(res_raw) else 0
        full_raw = full_raw[full_raw['method'].astype(str) != 'CARE-KM'].copy()
        res_raw = res_raw[res_raw['method'].astype(str) != 'CARE-KM'].copy()
        full_raw.to_csv(full_path, index=False)
        res_raw.to_csv(res_path, index=False)
        print(
            f'DERIVED-BUDGET CHECKPOINT MIGRATION: {dataset}: '
            f'recompute CARE-KM only (removed full={old_full_n}, resample={old_res_n}); '
            'published baseline rows retained.', flush=True
        )


    for method in METHODS:
        expected_runs = FULL_STOCHASTIC_REPS if method in el.STOCHASTIC_METHODS else 1
        done = set(
            full_raw.loc[full_raw['method'].astype(str) == method, 'run'].astype(int)
        ) if len(full_raw) else set()

        for run in range(expected_runs):
            if run in done:
                continue
            seed = BASE_SEED + run
            print(f'FULL {dataset:20s} | {method:10s} | run {run+1}/{expected_runs}', flush=True)
            row, _, _ = ck.run_method(
                method, X_all, y_all, k,
                seed=seed,
                care_config=CARE_CONFIG,
                si_sample_size=SI_SAMPLE_SIZE,
                si_seed=BASE_SEED + 999,
                max_iter=MAX_ITER,
            )
            row.update({
                'dataset': dataset,
                'run': int(run),
                'phase': 'full',
                'source_sha256': sha256_file(source_path),
            })
            full_raw = pd.concat([full_raw, pd.DataFrame([row])], ignore_index=True)
            full_raw.to_csv(full_path, index=False)


    for method in METHODS:
        expected_runs = FULL_STOCHASTIC_REPS if method in el.STOCHASTIC_METHODS else 1
        got = int(((full_raw['method'].astype(str) == method)).sum())
        if got != expected_runs:
            raise RuntimeError(f'{dataset}/{method}: full runs={got}, expected={expected_runs}')


    n_sub = int(round(len(X_all) * FRACTION))
    for rep in range(RESAMPLE_REPS):
        rng = np.random.default_rng(BASE_SEED + 10000 + rep)
        idx = np.sort(rng.choice(len(X_all), size=n_sub, replace=False))
        X = X_all[idx]
        y = y_all[idx]
        sample_hash = el.sample_index_hash(idx)

        done = set(
            res_raw.loc[pd.to_numeric(res_raw['rep'], errors='coerce') == rep, 'method'].astype(str)
        ) if len(res_raw) else set()

        for method in METHODS:
            if method in done:
                continue
            seed = BASE_SEED + 10000 + rep
            print(f'RESAMPLE {dataset:16s} | rep {rep+1:02d}/{RESAMPLE_REPS} | {method}', flush=True)
            row, _, _ = ck.run_method(
                method, X, y, k,
                seed=seed,
                care_config=CARE_CONFIG,
                si_sample_size=SI_SAMPLE_SIZE,
                si_seed=seed,
                max_iter=MAX_ITER,
            )
            row.update({
                'dataset': dataset,
                'rep': int(rep),
                'phase': 'resample',
                'sample_n': int(n_sub),
                'sample_hash': sample_hash,
                'source_sha256': sha256_file(source_path),
            })
            res_raw = pd.concat([res_raw, pd.DataFrame([row])], ignore_index=True)
            res_raw.to_csv(res_path, index=False)

    rep_counts = res_raw.groupby('rep')['method'].nunique()
    if len(rep_counts) != RESAMPLE_REPS or not (rep_counts == len(METHODS)).all():
        raise RuntimeError(f'{dataset}: resampling belum lengkap.')
    if not (res_raw.groupby('rep')['sample_hash'].nunique() == 1).all():
        raise RuntimeError(f'{dataset}: matched-resampling sample hash gagal.')

    fs = _full_summary(full_raw, dataset)
    rs = _resample_summary(res_raw, dataset)
    fs.to_csv(CHECKPOINT_DIR / f'{dataset}_full_summary.csv', index=False)
    rs.to_csv(CHECKPOINT_DIR / f'{dataset}_resample_summary.csv', index=False)

    all_full_raw.append(full_raw)
    all_full_summary.append(fs)
    all_resample_raw.append(res_raw)
    all_resample_summary.append(rs)

full_raw_all = pd.concat(all_full_raw, ignore_index=True)
full = pd.concat(all_full_summary, ignore_index=True)
resample_raw = pd.concat(all_resample_raw, ignore_index=True)
resample = pd.concat(all_resample_summary, ignore_index=True)

full_raw_all.to_csv(OUT_DIR / '10_cross_dataset_full_raw.csv', index=False)
full.to_csv(OUT_DIR / '10_cross_dataset_full.csv', index=False)
resample_raw.to_csv(OUT_DIR / '10_cross_dataset_resample_raw.csv', index=False)
resample.to_csv(OUT_DIR / '10_cross_dataset_resample_summary.csv', index=False)

print('\nGeneralization computation complete.')
print('Full summary rows:', len(full))
print('Resample summary rows:', len(resample))
print('Resample raw rows:', len(resample_raw))


protocol_hash = _sha256(LIB_DIR / 'care_km.py')
verified_protocols = []
for dataset in DATASETS:
    mp = CHECKPOINT_DIR / f'{dataset}_protocol.json'
    m = json.loads(mp.read_text(encoding='utf-8'))
    if m.get('care_km_sha256') != protocol_hash:
        raise RuntimeError(f'{dataset}: protocol belum terikat pada CARE-KM source hash saat ini.')
    fp = _load_checkpoint(CHECKPOINT_DIR / f'{dataset}_full_raw.csv', dataset, 'run')
    rp = _load_checkpoint(CHECKPOINT_DIR / f'{dataset}_resample_raw.csv', dataset, 'rep')
    if int((fp['method'].astype(str) == 'CARE-KM').sum()) != 1:
        raise RuntimeError(f'{dataset}: CARE-KM full-data row belum terverifikasi.')
    care_rep = rp[rp['method'].astype(str) == 'CARE-KM']
    if care_rep['rep'].nunique() != RESAMPLE_REPS:
        raise RuntimeError(f'{dataset}: CARE-KM resample rows belum lengkap.')
    verified_protocols.append(dataset)

stage10_stamp = {
    'stage': 10,
    'care_implementation_version': str(ck.CARE_IMPLEMENTATION_VERSION),
    'care_km_sha256': protocol_hash,
    'candidate_budget': int(CARE_CONFIG['candidate_budget']),
    'refinement_budget_rule': 'ceil(log2(candidate_budget))',
    'derived_refinement_budget': int(ck.derive_refinement_budget(CARE_CONFIG['candidate_budget'])),
    'datasets_verified': verified_protocols,
    'full_care_runs': int(len(DATASETS)),
    'resample_care_runs': int(len(DATASETS) * RESAMPLE_REPS),
    'published_baseline_checkpoints_reused': True,
}
(OUT_DIR / '10_DERIVED_BUDGET_EXECUTION.json').write_text(
    json.dumps(stage10_stamp, indent=2), encoding='utf-8'
)
print('Stage 10 derived-budget execution proof: PASS')
print(json.dumps(stage10_stamp, indent=2))


RANK_METRICS = {
    'dbi': 'lower',
    'si': 'higher',
    'final_cost': 'lower',
    'initial_cost': 'lower',
    'iterations': 'lower',
    'total_sec': 'lower',
}
RESAMPLE_COL = {
    'dbi': 'dbi_mean',
    'si': 'si_mean',
    'final_cost': 'final_cost_mean',
    'initial_cost': 'initial_cost_mean',
    'iterations': 'iterations_mean',
    'total_sec': 'total_sec_mean',
}

rank_rows = []
for dataset in DATASETS:
    gf = full[full['dataset'] == dataset].set_index('method').reindex(METHODS)
    gr = resample[resample['dataset'] == dataset].set_index('method').reindex(METHODS)

    if gf.isna().all(axis=1).any() or gr.isna().all(axis=1).any():
        raise RuntimeError(f'{dataset}: method tidak lengkap.')

    for metric, direction in RANK_METRICS.items():
        ascending = direction == 'lower'
        rf = pd.to_numeric(gf[metric], errors='raise').rank(
            method='average', ascending=ascending
        )
        rr = pd.to_numeric(gr[RESAMPLE_COL[metric]], errors='raise').rank(
            method='average', ascending=ascending
        )
        for method in METHODS:
            rank_rows.append({
                'dataset': dataset, 'source': 'full',
                'metric': metric, 'method': method,
                'rank': float(rf.loc[method]),
            })
            rank_rows.append({
                'dataset': dataset, 'source': 'resample_mean',
                'metric': metric, 'method': method,
                'rank': float(rr.loc[method]),
            })

ranks = pd.DataFrame(rank_rows)
average_rank = (
    ranks.groupby(['source','metric','method'], as_index=False)['rank']
    .mean().rename(columns={'rank':'average_rank'})
    .sort_values(['source','metric','average_rank'])
)
ranks.to_csv(OUT_DIR / '10_cross_dataset_ranks.csv', index=False)
average_rank.to_csv(OUT_DIR / '10_average_rank.csv', index=False)

wtl_rows = []
tol = 1e-12
for dataset in DATASETS:
    g = full[full['dataset'] == dataset].set_index('method').reindex(METHODS)
    for metric, direction in RANK_METRICS.items():
        care = float(g.loc['CARE-KM', metric])
        for baseline in [m for m in METHODS if m != 'CARE-KM']:
            ref = float(g.loc[baseline, metric])
            improvement = (ref-care) if direction == 'lower' else (care-ref)
            result = 'WIN' if improvement > tol else ('LOSS' if improvement < -tol else 'TIE')
            wtl_rows.append({
                'dataset': dataset,
                'metric': metric,
                'direction': direction,
                'baseline': baseline,
                'care_value': care,
                'baseline_value': ref,
                'improvement': improvement,
                'result': result,
            })

wtl = pd.DataFrame(wtl_rows)
wtl_summary = (
    wtl.groupby(['metric','baseline','result']).size()
    .unstack(fill_value=0).reset_index()
)
for c in ['WIN','TIE','LOSS']:
    if c not in wtl_summary.columns:
        wtl_summary[c] = 0

wtl.to_csv(OUT_DIR / '10_full_data_wtl.csv', index=False)
wtl_summary.to_csv(OUT_DIR / '10_full_data_wtl_summary.csv', index=False)

validation = {
    'datasets': int(len(DATASETS)),
    'dataset_keys': DATASETS,
    'methods': int(len(METHODS)),
    'full_stochastic_repetitions': int(FULL_STOCHASTIC_REPS),
    'resample_repetitions': int(RESAMPLE_REPS),
    'resample_fraction': float(FRACTION),
    'full_rows': int(len(full)),
    'resample_summary_rows': int(len(resample)),
    'resample_raw_rows': int(len(resample_raw)),
    'matched_resampling': True,
    'without_replacement': True,
    'clustering_rerun': True,
    'source_location': 'CARE-KM-PAPER/DATA',
}
(OUT_DIR / '10_generalization_validation.json').write_text(
    json.dumps(validation, indent=2), encoding='utf-8'
)

print(average_rank)
print(wtl_summary)
print(json.dumps(validation, indent=2))
print('STATUS: PASS')
