# Latar Belakang Peserta (Anonim)

Tabel ini memenuhi syarat **"anonymized user background"**. Tidak ada nama/identitas — hanya
kode P1–P5. Isi dari Bagian A Google Form. Skrip `analyze_user_testing.py` juga mencetak tabel
ini otomatis dari CSV (lihat `background_summary.png`).

| Peserta | Program studi | Tahun | Keakraban ML (1–5) | Keakraban dashboard (1–5) | Perangkat |
|---------|---------------|-------|--------------------|---------------------------|-----------|
| P1 | Ilmu Komputer | Tahun 3 | 4 | 4 | Laptop/Desktop |
| P2 | Desain Komunikasi Visual | Tahun 2 | 1 | 2 | HP |
| P3 | Teknik Industri | Tahun 4 | 3 | 3 | Laptop/Desktop |
| P4 | Akuntansi | Tahun 1 | 2 | 2 | HP |
| P5 | Sistem Informasi | Pascasarjana | 5 | 5 | Laptop/Desktop |

> Baris di atas adalah **contoh sintetis** yang cocok dengan `responses_template.csv` (untuk
> menguji pipeline). **Ganti** dengan data 5 peserta nyata sebelum dipakai di laporan.

**Usage scenario (ringkas untuk PPT):** tiap peserta membuka aplikasi yang sudah ter-deploy di
perangkat sendiri dan menyelesaikan 5 tugas nyata (memahami fungsi → membaca risk gauge →
mengubah input → upload CSV → menemukan recall/threshold), lalu mengisi kuesioner. Bukan
role-play; peserta mengevaluasi produk apa adanya. Detail tugas: `protokol-pengujian.md`.
