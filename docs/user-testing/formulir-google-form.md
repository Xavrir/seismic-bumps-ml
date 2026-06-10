# Spesifikasi Google Form — Pengujian Pengguna Seismic Risk Console

Salin pertanyaan di bawah ini ke **Google Forms** (forms.google.com). Tiap judul pertanyaan
**diawali kode dalam kurung siku** `[kode]` — biarkan kode itu di judulnya. Saat respons
diekspor ke CSV, header kolom akan memuat `[kode]`, dan `scripts/analyze_user_testing.py`
mencocokkan kolom lewat kode tersebut (jadi tahan terhadap perubahan teks/bahasa).

- **Skala Likert** = 1–5. Untuk SUS & kegunaan: **1 = Sangat tidak setuju, 5 = Sangat
  setuju**. Untuk kesulitan: **1 = Sangat mudah, 5 = Sangat sulit**. Untuk keakraban:
  **1 = Tidak familiar, 5 = Sangat familiar**.
- Atur **semua pertanyaan wajib (required)** kecuali yang terbuka boleh opsional.
- Di Google Forms, tipe **"Linear scale"** 1–5 untuk Likert; **"Multiple choice"** untuk
  pilihan; **"Short answer"** / **"Paragraph"** untuk teks.
- Estimasi pengisian: ±8–12 menit.

> Tip: aktifkan *Settings → Responses → Collect email addresses = OFF* (jaga anonimitas).

---

## Pembuka (deskripsi form)

> Terima kasih sudah membantu. Ini **demo riset**, bukan perangkat keselamatan tambang resmi.
> Kami menguji **aplikasinya**, bukan menguji Anda — tidak ada jawaban benar/salah. Form ini
> **anonim** (tanpa nama/email). Buka aplikasi di
> https://coalmine-seismic-risk.streamlit.app/ lalu kerjakan 5 tugas yang dipandu fasilitator
> sebelum mengisi.

---

## Bagian A — Latar belakang (anonim)

1. `[pid]` **Kode peserta** (diisi fasilitator, mis. P1) — *Short answer*
2. `[major]` **Program studi / jurusan** — *Short answer*
3. `[year]` **Tahun / jenjang kuliah** — *Multiple choice*: Tahun 1 / Tahun 2 / Tahun 3 /
   Tahun 4 / Pascasarjana
4. `[ml_fam]` **Seberapa familiar kamu dengan machine learning?** — *Linear scale 1–5*
   (1 = Tidak familiar, 5 = Sangat familiar)
5. `[dash_fam]` **Seberapa sering kamu memakai aplikasi data / dashboard?** —
   *Linear scale 1–5* (1 = Tidak pernah, 5 = Sangat sering)
6. `[device]` **Perangkat yang dipakai untuk tes ini** — *Multiple choice*:
   HP / Laptop atau Desktop / Tablet

## Bagian B — Tugas & tingkat kesulitan

> Untuk tiap tugas T1–T5 (lihat protokol): isi **status** dan **tingkat kesulitan**.

Status — *Multiple choice*: **Selesai / Sebagian / Gagal**
Kesulitan — *Linear scale 1–5* (1 = Sangat mudah, 5 = Sangat sulit)

7. `[t1_status]` **T1 — Memahami fungsi aplikasi dari panel "Start here": status**
8. `[t1_diff]` **T1: tingkat kesulitan**
9. `[t2_status]` **T2 — Membaca risk gauge, risk level, dan verdict di "Try one shift": status**
10. `[t2_diff]` **T2: tingkat kesulitan**
11. `[t3_status]` **T3 — Mengubah satu input lalu mengamati perubahan risiko: status**
12. `[t3_diff]` **T3: tingkat kesulitan**
13. `[t4_status]` **T4 — Upload CSV dan menemukan jumlah shift "dangerous": status**
14. `[t4_diff]` **T4: tingkat kesulitan**
15. `[t5_status]` **T5 — Menemukan recall model & arti threshold: status**
16. `[t5_diff]` **T5: tingkat kesulitan**

## Bagian C — Usability (SUS, 10 item, skala 1–5)

> Item berselang positif/negatif — dipakai menghitung skor SUS 0–100.

17. `[sus1]` Saya rasa saya ingin sering menggunakan aplikasi ini.
18. `[sus2]` Saya merasa aplikasi ini terlalu rumit.
19. `[sus3]` Saya rasa aplikasi ini mudah digunakan.
20. `[sus4]` Saya rasa saya butuh bantuan orang teknis untuk bisa memakai aplikasi ini.
21. `[sus5]` Saya rasa fitur-fitur aplikasi ini terintegrasi dengan baik.
22. `[sus6]` Saya rasa ada terlalu banyak ketidak-konsistenan dalam aplikasi ini.
23. `[sus7]` Saya rasa kebanyakan orang akan cepat belajar memakai aplikasi ini.
24. `[sus8]` Saya merasa aplikasi ini sangat janggal/membingungkan saat digunakan.
25. `[sus9]` Saya merasa percaya diri saat menggunakan aplikasi ini.
26. `[sus10]` Saya perlu belajar banyak hal dulu sebelum bisa memakai aplikasi ini.

## Bagian D — Usefulness / kegunaan (skala 1–5)

27. `[use1]` Saya paham arti **risk level** (low / watch / dangerous) yang ditampilkan.
28. `[use2]` Output aplikasi (skor & level risiko) terasa **berguna** untuk menilai bahaya shift.
29. `[use3]` Saya **percaya** hasil aplikasi ini layak dijadikan alat bantu pengambilan keputusan.
30. `[use4]` Tab **Model evidence / Methodology** membantu saya memahami **cara kerja & keterbatasan** model.
31. `[use5]` Secara keseluruhan, aplikasi ini menjelaskan prediksinya dengan **transparan**.

## Bagian E — Umpan balik terbuka (kualitatif)

32. `[q_confuse]` Bagian mana yang paling **membingungkan** atau sulit? Kenapa? — *Paragraph*
33. `[q_like]` Apa yang kamu **sukai** dari aplikasi ini? — *Paragraph*
34. `[q_suggest]` **Saran perbaikan** apa yang kamu punya? — *Paragraph*
35. `[q_clarity]` Menurutmu, apakah output risiko sudah **jelas/mudah dipahami**? Jelaskan. — *Paragraph*

---

## Pemetaan kode → kolom CSV

Saat ekspor Google Forms, header memuat teks lengkap termasuk `[kode]`. Skrip analisis
mencari `[kode]` di tiap header. Kalau kamu lebih suka mengisi manual, gunakan
`responses_template.csv` yang sudah memakai **kode polos** sebagai header:

`pid, major, year, ml_fam, dash_fam, device, t1_status..t5_status, t1_diff..t5_diff,
sus1..sus10, use1..use5, q_confuse, q_like, q_suggest, q_clarity`

> Skrip menerima **dua-duanya**: header berupa kode polos (`sus1`) **atau** header yang
> mengandung `[sus1]`.
