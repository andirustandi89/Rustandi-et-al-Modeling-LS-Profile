# Menjalankan eksperimen

Letakkan isi proyek di `MyDrive/REVISED/` dengan struktur `DATA/`, `LIB/`, `PYTHON/`, dan `OUTPUT/`. Buka notebook dalam `PYTHON/` melalui Google Colab. Dependensi tercatat dalam `PYTHON/requirements.txt`.

## Tahap utama

`00_Jalankan_Pipeline_REVISED.ipynb` menjalankan tahap 01–24. Pengaturan bawaan:

```python
JALANKAN_PRAPROSES = False
JALANKAN_BENCHMARK = True
LANJUTKAN = True
```

Untuk menghitung praproses dari awal, ubah `JALANKAN_PRAPROSES = True`. Proses tersebut memerlukan dump SQL/arsip pada `DATA/mysql-export` dan `DATA/compiled_pendikom_moodle_logs.csv`. Benchmark memerlukan sepuluh arsip dataset yang namanya tercantum pada registry `LIB/generalization_data.py` dan notebook 31.

`LANJUTKAN=True` memakai keluaran tersimpan setelah input, kode, parameter, lingkungan, dan keluaran lolos pemeriksaan. Jika sumber berubah, mekanisme bawaan dapat mengarsipkan checkpoint lalu menghitung ulang tahap terkait. Pertahankan folder `OUTPUT`, termasuk `KONTROL`, `BANK`, dan model yang dirujuk tahap berikutnya.

## Tahap tambahan

Tahap 25–33 dibuka dan dijalankan satu per satu. Master tidak menjadwalkan tahap tersebut.

| Tahap | Prasyarat utama | Pekerjaan |
|---|---|---|
| 25 | Hasil praproses/split, LIB, dataset benchmark | Membandingkan delapan initializer dan menyimpan bank partisi |
| 26 | Tahap 25 dengan protokol penuh | Interpretasi semantik dan kesesuaian profil dengan ILS |
| 27 | Tahap 25–26 | Membandingkan operator pembentukan profil |
| 28 | Tahap 27 dengan protokol penuh; sumber classifier 22 | Membandingkan supervisi CDS dan ILS; menyimpan model |
| 29 | Tahap 25–27 dengan protokol penuh | Sensitivitas k = 8, 12, 16, 20, 24; acuan k=16 dan q=16 |
| 30 | Hasil utama dan tahap 25–28 | Waktu inference serta ketidaksepakatan direct assignment dan classifier |
| 31 | Tahap 25 dengan protokol penuh; sepuluh dataset benchmark asli | Runtime delapan metode dan audit kegagalan |
| 32 | Sumber tahap 02, 22, 25–27; data numerik/metadata LMS; compiled Pendikom | Validasi antarkursus dan temporal dengan fitting pada train masing-masing |
| 33 | Hasil penuh tahap 25–32 | Analisis gabungan dari hasil tersimpan |

Keluaran tambahan berada pada `OUTPUT/TAMBAHAN/<nama_tahap>/`. Catatan per-run tahap 25 berada pada `OUTPUT/TAMBAHAN/KONTROL/25_Perbandingan_Initializer_Terintegrasi/`.

Label ILS digunakan sesuai protokol setiap tahap. Identitas peserta, metadata split, dan label referensi tidak menjadi fitur clustering. Protokol rinci, definisi metrik, dan kebijakan unresolved tetap tercantum dalam notebook serta berkas hasil.

## Berkas yang perlu tersedia di Drive

Arsip yang diberikan untuk perapian ini memuat LIB, notebook, dan hasil tersimpan, tetapi tidak menyertakan input mentah DATA. Gunakan DATA asli yang sudah ada di Drive untuk praproses, pengukuran benchmark baru, atau rekonstruksi raw Pendikom pada tahap 32. Nama dan isi sumber harus sesuai pemeriksaan bawaan proyek.

Judul notebook tetap tersedia; pengantar berulang telah dipindahkan ke petunjuk ini. Kode, konfigurasi, dan dependensi proyek menggunakan versi yang sama dengan arsip sumber.
