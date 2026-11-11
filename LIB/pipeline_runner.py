"""Urutan praproses, pembagian data, eksperimen, tabel, dan gambar."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,re
from pipeline_paths import find_root,configure,artifact_path,lms_data,pendikom_data
from pipeline_data import preflight
from pipeline_cache import (CACHE_SCHEMA,FileHashes,input_state,inventory,output_roots,runtime_signature,cache_decision,quarantine,save_status,digest_json)
import os
STAGES=['03_Lingkungan_dan_Reproduksibilitas', '05_Verifikasi_Formulasi_CARE_KM', '06_Verifikasi_Inisialisasi_Baseline', '07_Eksperimen_LMS_Full', '08_Lintasan_Refinement_CARE_KM', '09_Robustness_LMS_30x80', '10_Statistik_LMS', '11_Interpretasi_Semantik_FSLSM', '12_Sensitivitas_Aturan_Semantik', '13_Stabilitas_Semantik', '14_Stabilitas_Partisi', '15_Sensitivitas_Praproses', '16_Sensitivitas_Candidate_Budget', '17_Ablasi_Komponen_CARE_KM', '18_Audit_Dataset_Benchmark', '19_Generalisasi_Benchmark', '20_Statistik_Benchmark', '21_Runtime_Stabilitas_Skalabilitas', '22_Holdout_LMS_Pendikom', '23_Tabel_dan_Gambar', '24_Gambar_Reviewer']
BENCHMARK={'19_Generalisasi_Benchmark', '24_Gambar_Reviewer', '21_Runtime_Stabilitas_Skalabilitas', '23_Tabel_dan_Gambar', '20_Statistik_Benchmark', '18_Audit_Dataset_Benchmark'}
EXPECTED={'03_Lingkungan_dan_Reproduksibilitas': ['00_dataset_audit.csv', '00_protocol.json'], '05_Verifikasi_Formulasi_CARE_KM': ['02_formulation_verification.csv'], '06_Verifikasi_Inisialisasi_Baseline': ['03_baseline_initialization_verification.csv'], '07_Eksperimen_LMS_Full': ['04_full_dataset_summary.csv', '04_carekm_reference_labels.csv', '04_carekm_reference_final_modes.csv'], '08_Lintasan_Refinement_CARE_KM': ['05_refinement_trajectory_summary.csv'], '09_Robustness_LMS_30x80': ['06_robustness_raw.csv', '06_robustness_summary.csv'], '10_Statistik_LMS': ['07_wilcoxon_holm.csv'], '11_Interpretasi_Semantik_FSLSM': ['08_cluster_to_fslsm_mapping.csv', '08_theory_guided_pseudolabeled_dataset.csv'], '12_Sensitivitas_Aturan_Semantik': ['semantic_rule_sensitivity_summary.csv'], '13_Stabilitas_Semantik': ['semantic_stability_member_comparison.csv'], '14_Stabilitas_Partisi': ['partition_stability_summary.csv'], '15_Sensitivitas_Praproses': ['preprocessing_sensitivity_summary.csv'], '16_Sensitivitas_Candidate_Budget': ['18_Q_SENSITIVITY/18_q_sensitivity_summary.csv'], '17_Ablasi_Komponen_CARE_KM': ['19_COMPONENT_ABLATION/19_component_ablation_summary.csv'], '18_Audit_Dataset_Benchmark': ['09_generalization_dataset_audit.csv'], '19_Generalisasi_Benchmark': ['10_cross_dataset_full.csv', '10_cross_dataset_resample_summary.csv', '10_cross_dataset_resample_raw.csv'], '20_Statistik_Benchmark': ['11_friedman_kendall.csv', '11_cross_dataset_pairwise_wilcoxon_holm.csv'], '21_Runtime_Stabilitas_Skalabilitas': ['12_resample_runtime_scalability.csv'], '22_Holdout_LMS_Pendikom': ['70_30_main_lms_prospective_dimensional_accuracy_numeric.csv', '70_30_main_lms_prospective_predictions.csv', '70_30_groundtruth_classifier_predictions.csv', '70_30_table11_dimensional_performance_vs_ils_numeric.csv', '70_30_main_lms_classifier_metrics.csv'], '23_Tabel_dan_Gambar': ['13_table_07_generalization_average_rank.csv', '13_paper_artifact_manifest.json', '13_table_main_lms_classification.csv'], '24_Gambar_Reviewer': ['FIGURES_REVIEWER/reviewer_fig_02_cross_dataset_average_rank_bar.png'], '01_Praproses_main_LMS': [], '02_Praproses_Pendikom': [], '04_Split_LMS_Pendikom': []}

PREPROCESS=['01_Praproses_main_LMS','02_Praproses_Pendikom']
def required_outputs(stem,root):
    if stem==PREPROCESS[0]:return [lms_data(key,root) for key in ('selected','numeric','lmh','codes','categorical')]
    if stem==PREPROCESS[1]:return [pendikom_data(key,root) for key in ('numeric','categorical')]
    if stem=='04_Split_LMS_Pendikom':
        return [root/'OUTPUT/SPLIT'/dataset/part/'categorical.csv' for dataset in ('LMS','PENDIKOM') for part in ('TRAIN','TEST')]
    return [artifact_path(name,stem,root) for name in EXPECTED.get(stem,[])]

def check_inputs(root,preprocess,benchmarks):
    missing=[]
    if preprocess:
        if not any(p.is_file() for p in (root/'DATA').glob('mysql-export*')):missing.append('DATA/mysql-export (SQL atau arsip)')
        if not (root/'DATA/compiled_pendikom_moodle_logs.csv').is_file():missing.append('DATA/compiled_pendikom_moodle_logs.csv')
    else:
        preflight(root,benchmarks=False)
    if benchmarks:
        import generalization_data as gd
        missing += ['DATA/'+gd.DATASETS[key]['file'] for key in gd.DATASET_ORDER if not (root/'DATA'/gd.DATASETS[key]['file']).is_file()]
    if missing:raise FileNotFoundError('Lengkapi masukan proyek sebelum menjalankan pipeline:\n'+'\n'.join(missing))

def notebook_order():
    return PREPROCESS+[STAGES[0],'04_Split_LMS_Pendikom']+STAGES[1:]

def validate_notebook_inventory(root):
    scheduled=notebook_order()
    actual={path.stem for path in (root/'PYTHON').glob('*.ipynb')}
    expected=set(scheduled)|{'00_Jalankan_Pipeline_REVISED'}
    if len(scheduled)!=len(set(scheduled)):
        raise RuntimeError('Daftar tahap memiliki notebook berulang.')
    if actual!=expected:
        raise RuntimeError('Daftar notebook belum sesuai paket. Kurang: '+str(sorted(expected-actual))+
                           '; tambahan: '+str(sorted(actual-expected))+'. Gunakan folder PYTHON dari paket terbaru secara utuh.')
    if any(not re.match(r'^\d{2}_',name) for name in actual):raise RuntimeError('Notebook harus diawali nomor dua digit.')
    numbers=sorted(int(name.split('_',1)[0]) for name in actual)
    if numbers!=list(range(len(expected))):raise RuntimeError('Nomor notebook harus unik dan berurutan dari 00.')
    return scheduled

def run_pipeline(root=None,preprocess=True,benchmarks=True,resume=True,stages=None):
    root=configure(find_root(root))
    scheduled=validate_notebook_inventory(root)
    target=scheduled if preprocess else [stem for stem in scheduled if stem not in PREPROCESS]
    if stages is not None:target=list(stages)
    if not benchmarks:target=[name for name in target if name not in BENCHMARK]
    check_inputs(root,preprocess if stages is None else any(s in PREPROCESS for s in target),benchmarks)
    notebooks={}
    for stem in target:
        path=root/'PYTHON'/(stem+'.ipynb')
        notebooks[stem]=json.loads(path.read_text(encoding='utf-8-sig'))
        for c in notebooks[stem]['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),str(path),'exec')
    control=root/'OUTPUT/KONTROL';control.mkdir(parents=True,exist_ok=True)
    manifest=control/'status_pipeline.json'
    saved=json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else []
    status={row['notebook']:row for row in saved}
    hashes=FileHashes()
    runtime=runtime_signature()
    options={'preprocess':preprocess,'benchmarks':benchmarks}
    execution_inputs=None
    for number,stem in enumerate(target,1):
        print(f'[{number}/{len(target)}] Memeriksa isi input dan cache: {stem}',flush=True)
        expected=required_outputs(stem,root)
        path=root/'PYTHON'/(stem+'.ipynb')
        notebooks[stem]=json.loads(path.read_text(encoding='utf-8-sig'))
        current=input_state(stem,root,hashes,runtime,options)
        executable=[''.join(c['source']) for c in notebooks[stem]['cells'] if c['cell_type']=='code']
        if digest_json(executable)!=current['manifest']['notebooks'][path.relative_to(root).as_posix()]:
            raise RuntimeError('Notebook berubah saat disiapkan. Jalankan master kembali dengan kode yang tetap.')
        stable_keys=('data','libraries','notebooks','requirements','parameters','runtime','options')
        stable=digest_json({k:current['manifest'][k] for k in stable_keys})
        if execution_inputs is None:execution_inputs=stable
        elif execution_inputs!=stable:
            raise RuntimeError('Data mentah, kode, atau parameter berubah di tengah pipeline. Jalankan master kembali.')
        previous=status.get(stem,{})
        outputs=inventory(root,output_roots(stem,root),hashes)
        decision,reason=cache_decision(previous,current,outputs,resume)
        if decision=='skip' and all(p.is_file() and p.stat().st_size for p in expected):
            print(f'[{number}/{len(target)}] Lewati, isi terverifikasi sama: {stem}',flush=True)
            continue
        print(f'\n[{number}/{len(target)}] Jalankan: {stem} ({reason})',flush=True)
        record={'notebook':stem,'status':'berjalan','phase':'preparation',
                'mulai_utc':datetime.now(timezone.utc).isoformat(),
                'cache_schema':CACHE_SCHEMA,'input_sha256':current['sha256'],
                'input_manifest':current['manifest'],'alasan':reason}
        status[stem]=record
        save_status(manifest,status)
        previous_rebuild=os.environ.get('CARE_KM_REBUILD_PREPROCESS')
        try:
            if decision!='partial':record['cadangan']=quarantine(stem,root)
            if stem in PREPROCESS:os.environ['CARE_KM_REBUILD_PREPROCESS']='1'
            record['phase']='execution'
            save_status(manifest,status)
            for module in ('care_km','experiment_lib','generalization_data'):sys.modules.pop(module,None)
            scope={'__name__':'__main__','__file__':str(path)}
            for i,c in enumerate(notebooks[stem]['cells'],1):
                if c['cell_type']=='code':exec(compile(''.join(c['source']),f'{path.name}:sel{i}','exec'),scope)
            missing=[str(p) for p in expected if not p.is_file() or not p.stat().st_size]
            if missing:raise RuntimeError('Keluaran tahap belum lengkap: '+', '.join(missing))
            after=input_state(stem,root,hashes,runtime,options)
            if after['sha256']!=current['sha256']:
                record['phase']='invalidated'
                raise RuntimeError('Masukan atau kode berubah selama eksekusi tahap. Jalankan ulang dengan input yang tetap.')
            record.update(status='selesai',phase='finished',
                          output_manifest=inventory(root,output_roots(stem,root),hashes),
                          keluaran=[str(p.relative_to(root)) for p in expected])
        except BaseException as error:
            record.update(status='gagal',error=f'{type(error).__name__}: {error}')
            if record['phase']=='execution':
                try:
                    failure_state=input_state(stem,root,hashes,runtime,options)
                    if failure_state['sha256']!=current['sha256']:record['phase']='invalidated'
                    record['output_manifest']=inventory(root,output_roots(stem,root),FileHashes())
                except Exception as cache_error:
                    record['phase']='invalidated'
                    record['cache_diagnostic_error']=str(cache_error)
            raise
        finally:
            if previous_rebuild is None:os.environ.pop('CARE_KM_REBUILD_PREPROCESS',None)
            else:os.environ['CARE_KM_REBUILD_PREPROCESS']=previous_rebuild
            record['akhir_utc']=datetime.now(timezone.utc).isoformat()
            save_status(manifest,status)
    print('Pipeline selesai. Tabel dan gambar tersedia pada OUTPUT/TABEL_DAN_GAMBAR.')
    return [status[stem] for stem in target]
