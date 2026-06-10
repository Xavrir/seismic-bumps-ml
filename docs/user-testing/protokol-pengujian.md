# Protokol Pengujian Pengguna — Seismic Risk Console

Dokumen ini adalah rencana resmi *real user testing* untuk memenuhi syarat wajib
COMP6577001 (Machine Learning Final Project): minimal **5 pengguna nyata di luar tim**,
dengan latar belakang anonim, skenario penggunaan, serta umpan balik **usability** dan
**usefulness** (kuantitatif + kualitatif) yang dianalisis dan dipresentasikan di PPT.

- **Produk yang diuji:** Seismic Risk Console (aplikasi web Streamlit).
- **URL langsung:** https://coalmine-seismic-risk.streamlit.app/
- **Versi:** redesign "Tactical Telemetry" (risk gauge + verdict, kartu batch, grafik gelap).
- **Bahasa pengujian:** Bahasa Indonesia.

---

## 1. Tujuan

Mengukur seberapa **mudah digunakan** (usability) dan seberapa **berguna/ dapat dipahami**
(usefulness) aplikasi ini bagi pengguna pertama kali, lewat tugas nyata pada aplikasi yang
sudah ter-deploy. **Bukan** menguji akurasi model, dan **bukan** role-play sebagai petugas
tambang — peserta memakai aplikasi apa adanya sebagai evaluator.

## 2. Kriteria & rekrutmen peserta

- **Jumlah:** minimal 5 orang. Disarankan 5–8.
- **Profil:** mahasiswa, **bukan anggota tim proyek**.
- **Tidak perlu** latar belakang pertambangan atau machine learning — justru baik bila
  beragam tingkat keakrabannya dengan ML/dashboard.
- **Perangkat:** boleh HP atau laptop/desktop (kita catat, karena layout berbeda).

## 3. Anonimisasi & persetujuan (etika)

- **Tidak mengumpulkan nama, NIM, email, atau identitas apa pun.** Setiap peserta diberi
  kode **P1–P5** saja.
- Partisipasi **sukarela**; peserta boleh berhenti kapan saja.
- Sampaikan di awal: *"Ini demo riset, bukan perangkat keselamatan tambang resmi. Kami
  menguji aplikasinya, bukan menguji Anda. Tidak ada jawaban benar atau salah."*
- Data hanya dipakai untuk laporan tugas kuliah.

## 4. Skenario penggunaan = 5 tugas nyata pada aplikasi

Peserta membuka aplikasi di perangkat sendiri dan menyelesaikan tugas berikut. Tugas-tugas
ini sekaligus menjadi **usage scenario** yang didokumentasikan.

| # | Tugas | Yang diamati |
|---|-------|--------------|
| **T1** | Buka aplikasi, baca panel **"Start here"** (3 langkah), lalu jelaskan dengan kata sendiri apa fungsi aplikasi ini. | Pemahaman awal / first impression |
| **T2** | Di tab **"Try one shift"**, pilih satu *example shift*, lalu baca **risk gauge** (skor /100), **risk level** (low/watch/dangerous), dan kalimat **verdict**-nya. | Keterbacaan output utama |
| **T3** | Buka panel **"Adjust monitoring inputs"**, ubah satu nilai (mis. naikkan **Total seismic energy / genergy** atau **Current shift energy / energy**), lalu amati apakah gauge & level berubah dan ke arah mana. | Pemahaman interaktivitas & sebab-akibat |
| **T4** | Di tab **"Upload CSV"**, unduh template, lalu unggah kembali (atau pakai file contoh), dan sebutkan **berapa shift yang "dangerous"** dari kartu ringkasan. | Alur batch & membaca ringkasan |
| **T5** | Di tab **"Model evidence"** / **"Methodology"**, temukan nilai **recall** model (≈0.577 / kartu "Lockbox recall") dan jelaskan arti **threshold** (0.080). | Kepercayaan & interpretabilitas |

> Catatan fasilitator: catat **berhasil/sebagian/gagal** dan kesulitan tiap tugas. Jangan
> membantu kecuali peserta benar-benar buntu >1 menit (catat bila membantu).

## 5. Metrik yang dikumpulkan

Diisi lewat **Google Form** (lihat `formulir-google-form.md`):

- **Latar belakang anonim** — program studi, tahun, keakraban ML (1–5), keakraban dashboard
  (1–5), perangkat.
- **Per-tugas (T1–T5)** — status penyelesaian (Selesai/Sebagian/Gagal) + tingkat kesulitan
  (1 = sangat mudah … 5 = sangat sulit).
- **Usability** — **SUS** (System Usability Scale) 10 item baku → skor 0–100.
- **Usefulness** — 5 item persepsi kegunaan (skala 1–5).
- **Kualitatif** — pertanyaan terbuka: paling membingungkan, yang disukai, saran, kejelasan
  output.

## 6. Prosedur sesi (per peserta, ±15 menit)

1. **(1 mnt)** Sampaikan tujuan + pernyataan anonim/sukarela (Bagian 3). Beri kode P*n*.
2. **(1 mnt)** Catat perangkat & konteks (latar belakang anonim — bisa lewat Bagian A form).
3. **(8 mnt)** Peserta mengerjakan **T1–T5** sambil *think-aloud* bila nyaman. Fasilitator
   mencatat status + kesulitan + kutipan menarik. **Jangan** mengarahkan jawaban.
4. **(4 mnt)** Peserta mengisi **Google Form** (Bagian B–E).
5. **(1 mnt)** Terima kasih; tanya satu hal terakhir: "Satu hal yang paling ingin kamu ubah?"

## 7. Output → PPT

Setelah ≥5 peserta selesai: ekspor respons Form ke CSV →
`python scripts/analyze_user_testing.py` → grafik & tabel di `reports/figures/user_testing/`
→ tempel ke `template-hasil-analisis.md` lalu ke slide *User Testing* di PPT.

## 8. Jadwal & alat

- **Alat:** browser + Google Form. Tidak ada instalasi untuk peserta.
- **Target:** kumpulkan 5+ respons, lalu analisis dalam satu sesi.

Lihat juga: `README.md` (alur lengkap), `formulir-google-form.md` (isi form),
`template-latar-belakang-peserta.md`, `template-hasil-analisis.md`.
