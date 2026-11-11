"""Sidik jari input dan keluaran untuk melanjutkan eksekusi proyek."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import uuid
from pipeline_paths import GROUPS

CACHE_SCHEMA = 1
PREPROCESS = ('01_Praproses_main_LMS', '02_Praproses_Pendikom')
HOLDOUT = '22_Holdout_LMS_Pendikom'
SPLIT = '04_Split_LMS_Pendikom'
DEPS = {
    '10_Statistik_LMS': ['07_Eksperimen_LMS_Full','08_Lintasan_Refinement_CARE_KM','09_Robustness_LMS_30x80'],
    '11_Interpretasi_Semantik_FSLSM': ['07_Eksperimen_LMS_Full'],
    '12_Sensitivitas_Aturan_Semantik': ['07_Eksperimen_LMS_Full','11_Interpretasi_Semantik_FSLSM'],
    '13_Stabilitas_Semantik': ['07_Eksperimen_LMS_Full','09_Robustness_LMS_30x80','11_Interpretasi_Semantik_FSLSM'],
    '14_Stabilitas_Partisi': ['09_Robustness_LMS_30x80','13_Stabilitas_Semantik'],
    '15_Sensitivitas_Praproses': ['07_Eksperimen_LMS_Full','11_Interpretasi_Semantik_FSLSM'],
    '19_Generalisasi_Benchmark': ['18_Audit_Dataset_Benchmark'],
    '20_Statistik_Benchmark': ['19_Generalisasi_Benchmark'],
    '21_Runtime_Stabilitas_Skalabilitas': ['18_Audit_Dataset_Benchmark','19_Generalisasi_Benchmark'],
    HOLDOUT: [SPLIT],
    '23_Tabel_dan_Gambar': ['07_Eksperimen_LMS_Full','08_Lintasan_Refinement_CARE_KM','09_Robustness_LMS_30x80',
        '10_Statistik_LMS','11_Interpretasi_Semantik_FSLSM','18_Audit_Dataset_Benchmark',
        '19_Generalisasi_Benchmark','20_Statistik_Benchmark',
        '21_Runtime_Stabilitas_Skalabilitas','16_Sensitivitas_Candidate_Budget','17_Ablasi_Komponen_CARE_KM',HOLDOUT],
    '24_Gambar_Reviewer': ['09_Robustness_LMS_30x80','23_Tabel_dan_Gambar'],
}

def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()

class FileHashes:
    """Baca isi lagi pada setiap pemanggilan master; gunakan ulang hash dalam satu eksekusi."""
    def __init__(self): self.cache = {}

    def file(self,path):
        path=Path(path)
        before=path.stat()
        stamp=(before.st_size,before.st_mtime_ns,before.st_ctime_ns,getattr(before,'st_ino',0))
        cached=self.cache.get(str(path))
        if cached and cached[0]==stamp:return cached[1]
        result=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):result.update(chunk)
        after=path.stat()
        if stamp!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns,getattr(after,'st_ino',0)):
            raise RuntimeError(f'Berkas berubah saat dibaca: {path}')
        record={'sha256':result.hexdigest(),'bytes':after.st_size}
        self.cache[str(path)]=(stamp,record)
        return record

def files_under(paths):
    result=set()
    for path in paths:
        path=Path(path)
        if path.is_file():result.add(path)
        elif path.is_dir():result.update(p for p in path.rglob('*') if p.is_file())
    return sorted(result,key=str)

def inventory(root,paths,hashes):
    return {p.relative_to(root).as_posix():hashes.file(p) for p in files_under(paths)
            if '__pycache__' not in p.parts and p.suffix!='.pyc'}

def output_roots(stem,root):
    output=root/'OUTPUT'
    if stem in PREPROCESS:return [output/'PRAPROSES'/stem]
    if stem==SPLIT:return [output/'SPLIT/LMS',output/'SPLIT/PENDIKOM',output/'KONTROL/split_manifest.json']
    if stem==HOLDOUT:return [output/'EKSPERIMEN/HOLDOUT/LMS',output/'EKSPERIMEN/HOLDOUT/PENDIKOM']
    paths=[output/GROUPS[stem]]
    if stem in ('08_Lintasan_Refinement_CARE_KM','09_Robustness_LMS_30x80','13_Stabilitas_Semantik',
                '16_Sensitivitas_Candidate_Budget','17_Ablasi_Komponen_CARE_KM'):
        paths.append(output/'SPLIT/RESAMPLING/LMS'/stem)
    if stem=='19_Generalisasi_Benchmark':paths.append(output/'SPLIT/RESAMPLING/BENCHMARK'/stem)
    return paths

def runtime_signature():
    versions={}
    for module,distribution in [('numpy','numpy'),('pandas','pandas'),('scipy','scipy'),
          ('sklearn','scikit-learn'),('matplotlib','matplotlib'),('joblib','joblib'),('threadpoolctl','threadpoolctl')]:
        loaded=sys.modules.get(module)
        if loaded is not None and hasattr(loaded,'__version__'):versions[module]=loaded.__version__
        else:
            try:versions[module]=importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:versions[module]='tidak terpasang'
    return {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'packages':versions}

def input_state(stem,root,hashes,runtime,options):
    data=inventory(root,[root/'DATA'],hashes)
    libraries=inventory(root,[root/'LIB'],hashes)
    notebooks={}
    for p in sorted((root/'PYTHON').glob('*.ipynb')):
        nb=json.loads(p.read_text(encoding='utf-8-sig'))
        # Tampilan hasil dan hitungan eksekusi Colab tidak mengubah kode yang dijalankan.
        source=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        notebooks[p.relative_to(root).as_posix()]=digest_json(source)
    requirements=inventory(root,[root/'PYTHON/requirements.txt'],hashes)
    processed={} if stem in PREPROCESS else inventory(root,[root/'OUTPUT/PRAPROSES'],hashes)
    dependencies={}
    for dependency in DEPS.get(stem,[]):
        dependencies[dependency]=inventory(root,output_roots(dependency,root),hashes)
    environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('CARE_KM_')
                 and k not in ('CARE_KM_ROOT','CARE_KM_PAPER_ROOT','CARE_KM_REBUILD_PREPROCESS')}
    state={'schema':CACHE_SCHEMA,'root':str(root),'stage':stem,'data':data,'libraries':libraries,
           'notebooks':notebooks,'requirements':requirements,'preprocessed':processed,
           'dependencies':dependencies,'runtime':runtime,'parameters':environment,'options':options}
    return {'sha256':digest_json(state),'manifest':state}

def cache_decision(previous,current,outputs,resume):
    if not resume:return 'fresh','LANJUTKAN=False'
    if previous.get('cache_schema')!=CACHE_SCHEMA:return 'fresh','belum memiliki catatan isi yang terverifikasi'
    if previous.get('input_sha256')!=current['sha256']:return 'fresh','input, kode, parameter, lingkungan, atau sumber tahap berubah'
    saved=previous.get('output_manifest')
    if not saved or saved!=outputs:return 'fresh','isi atau daftar keluaran/checkpoint berubah'
    if previous.get('status')=='selesai':return 'skip','isi input dan seluruh keluaran identik'
    if previous.get('status')=='gagal' and previous.get('phase')=='execution':
        return 'partial','checkpoint parsial dan input masih identik'
    return 'fresh','tahap belum selesai dan checkpoint belum terverifikasi'

def _long_path(path):
    text=str(path)
    return Path('\\\\?\\'+text) if os.name=='nt' and not text.startswith('\\\\?\\') else Path(path)

def quarantine(stem,root):
    """Pisahkan keluaran lama agar checkpoint internal tidak memakai data yang sudah usang."""
    output=(root/'OUTPUT').resolve()
    token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    destination=output/'KONTROL/CACHE_CADANGAN'/token/stem
    moved=[]
    for path in output_roots(stem,root):
        if not path.exists():continue
        resolved=path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(output) or resolved==output:
            raise RuntimeError(f'Lokasi keluaran tidak aman untuk dipindahkan: {path}')
        if path.is_dir() and not any(path.iterdir()):continue
        target=destination/path.relative_to(root)
        if not target.resolve().is_relative_to(output/'KONTROL/CACHE_CADANGAN'):
            raise RuntimeError('Lokasi cadangan berada di luar folder proyek.')
        _long_path(target.parent).mkdir(parents=True,exist_ok=True)
        shutil.move(str(_long_path(path)),str(_long_path(target)))
        moved.append(target.relative_to(root).as_posix())
    return moved

def save_status(path,status):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(list(status.values()),indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)
