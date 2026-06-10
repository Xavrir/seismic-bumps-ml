# User Testing — Seismic Risk Console

Paket lengkap untuk *real user testing* (syarat wajib COMP6577001). Bahasa: Indonesia.
Diuji terhadap aplikasi ter-deploy: https://coalmine-seismic-risk.streamlit.app/

## Isi folder
| File | Fungsi |
|---|---|
| `protokol-pengujian.md` | Rencana resmi: tujuan, kriteria peserta, anonimisasi, **5 tugas (usage scenario)**, metrik, prosedur sesi. |
| `formulir-google-form.md` | Spesifikasi Google Form siap-salin (Bagian A–E, 35 pertanyaan termasuk SUS 10 item). |
| `template-latar-belakang-peserta.md` | Tabel latar belakang anonim P1–P5. |
| `responses_template.csv` | Header kolom + 5 baris **contoh sintetis** (untuk uji pipeline; ganti dengan data nyata). |
| `template-hasil-analisis.md` | Kerangka hasil yang dipetakan ke slide PPT. |
| `analysis_summary.md`, `qualitative_responses.md` | **Dihasilkan** oleh skrip analisis. |

Skrip: `../../scripts/analyze_user_testing.py` → grafik ke `../../reports/figures/user_testing/`.

## Alur end-to-end
1. **Baca** `protokol-pengujian.md`.
2. **Bangun Google Form** dari `formulir-google-form.md` (pertahankan prefix `[kode]` di tiap judul).
3. **Rekrut 5 mahasiswa** (bukan tim). Per peserta: jalankan 5 tugas, lalu isi form. Beri kode P1–P5.
4. **Ekspor** respons Form → CSV.
5. **Analisis:**
   ```bash
   # dari root repo, pakai venv yang punya pandas + matplotlib
   python scripts/analyze_user_testing.py --input <responses.csv>
   ```
   Tanpa `--input`, skrip memakai `responses_template.csv` (data contoh) untuk demo.
6. **Susun PPT** dari `analysis_summary.md`, `qualitative_responses.md`, dan PNG di
   `reports/figures/user_testing/`, mengikuti `template-hasil-analisis.md`.

## Catatan
- Skrip mencocokkan kolom via kode (`sus1`) **atau** header yang memuat `[sus1]`, jadi ekspor
  Google Forms mentah bisa langsung dipakai bila judul diberi prefix kode.
- SUS = 0–100 (sudah diverifikasi: semua-terbaik → 100, semua-terburuk → 0).
- Anonim: **tanpa nama/NIM/email**; hanya kode P1–P5.
