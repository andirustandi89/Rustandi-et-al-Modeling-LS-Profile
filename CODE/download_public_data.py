# Mengunduh dataset publik PENDIKOM dan benchmark UCI dari sumber asal.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import argparse
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PENDIKOM = ROOT / 'DATA' / 'PENDIKOM'
BENCH = ROOT / 'DATA' / 'BENCHMARKS'

PENDIKOM_URL = 'https://raw.githubusercontent.com/Dewi2501/Learning-Style-Based/main/DATA/compiled_pendikom_moodle_logs.csv'

BENCHMARKS = {
    'dataset_car+evaluation.zip': ('https://archive.ics.uci.edu/static/public/19/car+evaluation.zip', '1559d51dcf327f4f8c71b711ceed7fd95a382fc8d8e1998667f4f23b82860403'),
    'dataset_lymphography.zip': ('https://archive.ics.uci.edu/static/public/63/lymphography.zip', '082ff211040ad7dc2122e92441aa3eee494ab6496fecdcd86a440fe863ee5b6f'),
    'hayes+roth.zip': ('https://archive.ics.uci.edu/static/public/44/hayes+roth.zip', 'c034173ccd2f04e887afa8688be86ec10d72f2f1e8481e967ac60d8d04a711f1'),
    'statlog+australian+credit+approval.zip': ('https://archive.ics.uci.edu/static/public/143/statlog+australian+credit+approval.zip', 'f0bd9cd9b1d133cb20456b8953e1f8139fd30ed2b3f66da540db75136f1e9e0b'),
    'chronic+kidney+disease.zip': ('https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip', '3f0d0e5bab0f453165acfa2c77eae695393089c5b338cb4cd98339c96a26a585'),
    'annealing.zip': ('https://archive.ics.uci.edu/static/public/3/annealing.zip', '75bcc8f872dda28a0fb2954235eb53fc9d6e34f0115577199fddd5a1583251a7'),
    'statlog+heart.zip': ('https://archive.ics.uci.edu/static/public/145/statlog+heart.zip', '04e44018af4f4b32008029fbfb1079e112d3b372bb032dd1ae361c74d7528b2c'),
    'heart+disease.zip': ('https://archive.ics.uci.edu/static/public/45/heart+disease.zip', 'b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc'),
    'thyroid+disease.zip': ('https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip', 'a0982569a7442c03a20815db58f271245e7a111b10ac46f6c6b5fa6feee4c1f4'),
    'dataset_nursery.zip': ('https://archive.ics.uci.edu/static/public/76/nursery.zip', '914780f5e3050895216cb91f9e4a1408f3c0813b5854d5a6d6c601c2882091dd'),
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ambil(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print('Unduh:', url)
    urllib.request.urlretrieve(url, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pendikom-only', action='store_true')
    parser.add_argument('--benchmarks-only', action='store_true')
    args = parser.parse_args()

    if not args.benchmarks_only:
        target = PENDIKOM / 'compiled_pendikom_moodle_logs.csv'
        if not target.exists():
            ambil(PENDIKOM_URL, target)
        print('PENDIKOM:', target)

    if not args.pendikom_only:
        for name, (url, expected) in BENCHMARKS.items():
            target = BENCH / name
            if not target.exists():
                ambil(url, target)
            actual = sha256(target)
            if actual != expected:
                raise RuntimeError(f'Checksum berbeda untuk {name}: {actual}')
            print('OK:', name)

if __name__ == '__main__':
    main()
