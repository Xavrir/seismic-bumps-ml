# User Testing — Seismic Risk Console

Paket lengkap untuk *real user testing* (syarat wajib COMP6577001). Bahasa: Indonesia.
Diuji terhadap aplikasi ter-deploy: https://coalmine-seismic-risk.streamlit.app/

## Isi folder
| File | Fungsi |
|---|---|
| `protokol-pengujian.md` | Rencana resmi: tujuan, kriteria peserta, anonimisasi, **5 tugas (usage scenario)**, metrik, prosedur sesi. |
| `formulir-google-form.md` | Spesifikasi Google Form siap-salin (Bagian A–E, 35 pertanyaan termasuk SUS 10 item). |
| `create_form.gs` | **Google Apps Script** yang membangun seluruh form otomatis (cara tercepat). |
| `template-latar-belakang-peserta.md` | Tabel latar belakang anonim P1–P5. |
| `responses_template.csv` | Header kolom + 5 baris **contoh sintetis** (untuk uji pipeline; ganti dengan data nyata). |
| `template-hasil-analisis.md` | Kerangka hasil yang dipetakan ke slide PPT. |
| `analysis_summary.md`, `qualitative_responses.md` | **Dihasilkan** oleh skrip analisis. |

Skrip: `../../scripts/analyze_user_testing.py` → grafik ke `../../reports/figures/user_testing/`.

## Buat form otomatis (cara tercepat — Apps Script)
1. Buka https://script.google.com → **New project**.
2. Tempel seluruh isi `create_form.gs` (ganti `Code.gs` bawaan).
3. Jalankan fungsi **`createSeismicUserTestingForm`** (izinkan akses saat diminta — sekali saja).
4. Buka **View → Logs** untuk melihat **Edit URL** dan **Live URL** form.
5. Bagikan **Live URL** ke peserta. Form sudah anonim, lengkap 35 pertanyaan dengan prefix `[kode]`.

> Alternatif manual: bangun form dengan menyalin `formulir-google-form.md`.

## Alur end-to-end
1. **Baca** `protokol-pengujian.md`.
2. **Buat Google Form** — jalankan `create_form.gs` (atau salin manual dari `formulir-google-form.md`).
3. **Rekrut 5 mahasiswa** (bukan tim). Per peserta: jalankan 5 tugas, lalu isi form. Beri kode P1–P5.
4. **Ekspor** respons Form → CSV (tab Responses → "..." → Download responses .csv).
5. **Analisis:**
   ```bash
   # dari root repo, pakai venv yang punya pandas + matplotlib
   python scripts/analyze_user_testing.py --input <responses.csv>
   ```
   Tanpa `--input`, skrip memakai `responses_template.csv` (data contoh) untuk demo.
6. **Susun PPT** dari `analysis_summary.md`, `qualitative_responses.md`, dan PNG di
   `reports/figures/user_testing/`, mengikuti `template-hasil-analisis.md`.

## Pesan rekrutmen (siap kirim ke grup/chat)
> Halo! Aku lagi mengerjakan tugas Machine Learning dan butuh **5–6 orang** untuk mencoba
> sebuah web app singkat (±15 menit, dari HP/laptop). Tugasnya cuma pakai aplikasinya lalu
> isi kuesioner singkat. **Anonim** (tanpa nama/email), dan kamu **bukan** bagian tim
> proyekku ya. Tertarik bantu? Nanti aku pandu langkahnya. Makasih banyak 🙏

## Catatan
- Skrip mencocokkan kolom via kode (`sus1`) **atau** header yang memuat `[sus1]`, jadi ekspor
  Google Forms mentah bisa langsung dipakai bila judul diberi prefix kode.
- SUS = 0–100 (sudah diverifikasi: semua-terbaik → 100, semua-terburuk → 0).
- Anonim: **tanpa nama/NIM/email**; hanya kode P1–P5.
- **Wajib pengguna nyata** (≥5, di luar tim). `responses_template.csv` hanya data sintetis untuk
  menguji skrip — jangan dipakai sebagai hasil pengujian di PPT.
