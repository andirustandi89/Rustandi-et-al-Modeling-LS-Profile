from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

FROZEN_CARE_CONFIG = {
    "candidate_rule": "separation",
    "candidate_budget": 16,
    "min_gain": 0.0,
}


def frozen_care_metadata():
    q = int(FROZEN_CARE_CONFIG["candidate_budget"])
    derived = int(np.ceil(np.log2(q))) if q > 1 else 0
    return {
        **FROZEN_CARE_CONFIG,
        "parent_initializer": "Cao",
        "refinement_budget_rule": "ceil(log2(candidate_budget))",
        "derived_refinement_budget": derived,
        "exchange_depth_is_independent_hyperparameter": False,
    }

K = 16
MAX_ITER = 100
SI_SAMPLE_SIZE = 2000
BASE_SEED = 161600

METHODS = [
    "Random", "Huang", "Cao", "Bai",
    "K-Modes++", "HR-2023", "CARE-KM",
]
STOCHASTIC_METHODS = {"Random", "Huang", "K-Modes++", "HR-2023"}
DETERMINISTIC_METHODS = {"Cao", "Bai", "CARE-KM"}

EXPECTED_FEATURES = ['content_visit', 'content_stay', 'selfass_visit_proxy', 'selfass_stay_proxy', 'exercise_visit_proxy', 'exercise_stay_proxy', 'forum_visit', 'forum_stay', 'forum_post', 'quiz_stay_results_proxy', 'navigation_overview_visit', 'navigation_overview_stay', 'return_to_overview_rate']


def mount_drive_if_colab():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return True
    except Exception:
        return False


def find_paper_root(start=None):
    from pipeline_paths import find_root
    return find_root()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(df, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, float_format="%.6f")


def load_lms_dataset(root, filename="categorical"):
    root = Path(root)
    from pipeline_paths import lms_data
    path = lms_data(filename, root)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
    if "user" not in df.columns:
        raise RuntimeError("Kolom user tidak ditemukan.")
    features = [c for c in df.columns if c != "user"]
    if features != EXPECTED_FEATURES:
        raise RuntimeError(f"Feature set/order berbeda. Expected={EXPECTED_FEATURES}, actual={features}")
    if df["user"].duplicated().any():
        raise RuntimeError("User ID duplikat ditemukan.")
    invalid = {}
    for c in features:
        vals = set(df[c].astype(str).str.strip().unique())
        if not vals <= {"1", "2", "3"}:
            invalid[c] = sorted(vals - {"1", "2", "3"})
    if invalid:
        raise RuntimeError(f"Kategori di luar 1/2/3: {invalid}")
    X = df[features].astype(str).to_numpy(dtype=object)
    return df, X, features


def dataset_audit(df, features):
    return pd.DataFrame([{
        "rows": int(len(df)),
        "unique_users": int(df["user"].nunique()),
        "features": int(len(features)),
        "duplicate_users": int(df["user"].duplicated().sum()),
        "missing_cells": int((df[features].astype(str) == "").sum().sum()),
        "min_cardinality": int(min(df[c].nunique() for c in features)),
        "max_cardinality": int(max(df[c].nunique() for c in features)),
        "all_values_in_1_2_3": bool(all(set(df[c].unique()) <= {"1","2","3"} for c in features)),
    }])


def summarize_runs(raw):
    metrics = [
        "initial_cost", "final_cost", "dbi", "si", "iterations",
        "min_cluster", "max_cluster", "cluster_cv",
        "init_sec", "cluster_sec", "total_sec",
    ]
    agg = {}
    for m in metrics:
        if m in raw.columns:
            agg[m] = (m, "mean")
            agg[m + "_sd"] = (m, "std")
    out = raw.groupby("method").agg(n=("method", "size"), **agg).reset_index()
    return out


def sample_index_hash(indices):
    a = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, pos in enumerate(order):
        value = (m - rank) * p[pos]
        running = max(running, value)
        adjusted[pos] = min(1.0, running)
    return adjusted


def main_lms_classifier_metrics(root):
    from pipeline_paths import artifact_path
    from sklearn.metrics import balanced_accuracy_score,f1_score,accuracy_score
    pred=pd.read_csv(artifact_path('70_30_main_lms_prospective_predictions.csv',root=root))
    mapping=pd.read_csv(artifact_path('70_30_main_lms_frozen_cluster_to_fslsm_mapping.csv',root=root)).set_index('cluster_id')
    dims={'active_reflective':'Active / Reflective','sensing_intuitive':'Sensing / Intuitive',
          'visual_verbal':'Visual / Verbal','sequential_global':'Sequential / Global'}
    poles={'Act.':'Active','Ref.':'Reflective','Sens.':'Sensing','Int.':'Intuitive','Vis.':'Visual','Ver.':'Verbal','Seq.':'Sequential','Gl.':'Global'}
    rows=[]
    for model in ['NB','DT','SVM']:
        row={'model':model};acc=[];ba=[];f1=[]
        for col,label in dims.items():
            truth=pred['reference_cluster_id'].map(mapping[col]).map(poles).fillna('Balanced')
            prediction=pred[model+'_'+label]
            value=accuracy_score(truth,prediction)
            row[label]=value;acc.append(value)
            ba.append(balanced_accuracy_score(truth,prediction))
            f1.append(f1_score(truth,prediction,average='macro',zero_division=0))
        row.update({'Mean acc':np.mean(acc),'Mean BA':np.mean(ba),'Mean macro-F1':np.mean(f1)})
        rows.append(row)
    return pd.DataFrame(rows)
