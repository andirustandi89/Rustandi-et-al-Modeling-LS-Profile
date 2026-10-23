# Membentuk matriks kategorikal LMS utama dari matriks numerik learner-level.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'DATA' / 'MAIN_LMS'
NUMERIC_FILE = DATA_DIR / 'main_lms_numeric.csv'
CATEGORICAL_FILE = DATA_DIR / 'main_lms_categorical.csv'
EXPECTED_SHA256 = '57fef508915e617faf6a2c4dfb1b72f6a00f6ec008b8622f703d016503a9b319'

FEATURE_PLAN = [
    ('content_visit','graf',75.0,100.0),
    ('content_stay','empirical',None,None),
    ('selfass_visit_proxy','graf',25.0,75.0),
    ('selfass_stay_proxy','empirical',None,None),
    ('exercise_visit_proxy','graf',25.0,75.0),
    ('exercise_stay_proxy','empirical',None,None),
    ('forum_visit','degenerate',7.0,50.0),
    ('forum_stay','degenerate',5.0,30.0),
    ('forum_post','sparse',1.0,10.0),
    ('quiz_stay_results_proxy','graf',30.0,60.0),
    ('navigation_overview_visit','degenerate',10.0,20.0),
    ('navigation_overview_stay','empirical',None,None),
    ('return_to_overview_rate','empirical',None,None),
]

def potong(series, low, high):
    x = pd.to_numeric(series, errors='coerce').fillna(0).astype(float)
    return pd.Series(np.where(x < low, 1, np.where(x < high, 2, 3)), index=series.index, dtype='int8')

def kuantil_positif(series):
    x = pd.to_numeric(series, errors='coerce').fillna(0).astype(float)
    pos = x[x > 0]
    if pos.empty:
        raise RuntimeError('Tidak ada nilai positif untuk menentukan kuantil.')
    low = float(pos.quantile(1/3))
    high = float(pos.quantile(2/3))
    if high <= low:
        unik = np.sort(pos.unique())
        low = float(np.quantile(unik, 1/3))
        high = float(np.quantile(unik, 2/3))
    if high <= low:
        raise RuntimeError('Batas kuantil tidak valid.')
    return low, high

def proporsi(cat):
    return cat.value_counts(normalize=True).reindex([1,2,3], fill_value=0.0)

def main():
    data = pd.read_csv(NUMERIC_FILE, dtype={'user':str})
    hasil = pd.DataFrame({'user':data['user'].astype(str)})
    for fitur, mode, low, high in FEATURE_PLAN:
        x = data[fitur]
        if mode == 'graf':
            cat = potong(x, low, high)
        elif mode == 'empirical':
            low2, high2 = kuantil_positif(x)
            cat = potong(x, low2, high2)
        elif mode == 'degenerate':
            awal = potong(x, low, high)
            p = proporsi(awal)
            if float(p.max()) > 0.95 or int((p == 0).sum()) > 0:
                low2, high2 = kuantil_positif(x)
                cat = potong(x, low2, high2)
            else:
                cat = awal
        elif mode == 'sparse':
            awal = potong(x, low, high)
            p = proporsi(awal)
            if float(p.max()) > 0.95 or int((p == 0).sum()) > 0:
                xx = pd.to_numeric(x, errors='coerce').fillna(0).astype(float)
                pos = xx[xx > 0]
                median = float(pos.median())
                cat = pd.Series(np.where(xx <= 0, 1, np.where(xx < median, 2, 3)), index=x.index, dtype='int8')
            else:
                cat = awal
        hasil[fitur] = cat
    hasil.to_csv(CATEGORICAL_FILE, index=False)
    digest = hashlib.sha256(CATEGORICAL_FILE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f'Hash dataset kategorikal berbeda: {digest}')
    print(f'Dataset kategorikal siap: {CATEGORICAL_FILE}')
    print(f'Baris: {len(hasil)} | fitur: {len(hasil.columns)-1}')
    print(f'SHA256: {digest}')

if __name__ == '__main__':
    main()
