# Hasil Pengujian Pengguna — Template untuk PPT

Struktur ini langsung memetakan ke bagian **User Testing** di PPT. Angka & grafik diisi dari
`analysis_summary.md`, `qualitative_responses.md`, dan `reports/figures/user_testing/*.png`
(dihasilkan `scripts/analyze_user_testing.py`). Ganti placeholder `‹…›` dengan data nyata.

> Memenuhi syarat wajib: ≥5 pengguna nyata di luar tim, latar belakang anonim + skenario
> penggunaan, umpan balik usability + usefulness (kuantitatif + kualitatif), dianalisis &
> dipresentasikan.

---

## Slide 1 — Metodologi
- **Peserta:** ‹N› mahasiswa, bukan anggota tim, anonim (P1–P‹N›).
- **Metode:** task-based usability test pada aplikasi ter-deploy (bukan role-play).
- **Skenario:** tiap peserta menyelesaikan 5 tugas nyata (memahami fungsi → baca risk gauge →
  ubah input → upload CSV → temukan recall/threshold).
- **Instrumen:** Google Form — latar belakang, status & kesulitan per tugas, **SUS** (10 item),
  **usefulness** (5 item), 4 pertanyaan terbuka.
- **Alat analisis:** `scripts/analyze_user_testing.py`.

## Slide 2 — Latar belakang peserta (anonim)
Tempel tabel dari `template-latar-belakang-peserta.md` + grafik `background_summary.png`.

| Peserta | Prodi | Tahun | ML (1–5) | Dashboard (1–5) | Perangkat |
|---|---|---|---|---|---|
| P1 | ‹…› | ‹…› | ‹…› | ‹…› | ‹…› |
| … | | | | | |

## Slide 3 — Hasil kuantitatif: Usability (SUS)
- **Skor SUS rata-rata: ‹xx.x› / 100 → ‹grade›** (rujukan industri ≈ 68).
- Grafik: `sus_per_participant.png`.
- 1–2 kalimat interpretasi: ‹mis. "di atas rata-rata; usability dipersepsikan baik"›.

## Slide 4 — Hasil kuantitatif: Usefulness & tugas
- **Kegunaan rata-rata: ‹x.xx› / 5 (‹xx›%).** Grafik: `usefulness_items.png`.
- **Kesulitan & keberhasilan tugas:** grafik `task_metrics.png`.
- Soroti: tugas termudah ‹T?› dan tersulit ‹T?› (mis. T5 recall/threshold) + alasannya.

## Slide 5 — Temuan kualitatif (tema)
Dari `qualitative_responses.md`, rangkum jadi tema (sertakan 1–2 kutipan ringkas):
- **Disukai:** ‹tema, mis. "risk gauge & verdict jelas; desain konsisten"›.
- **Kendala:** ‹tema, mis. "istilah teknis (recall/threshold) sulit untuk awam; input numerik kecil di HP"›.
- **Saran:** ‹tema, mis. "mode penjelasan untuk awam; perbesar input di HP; ekspor PDF"›.

## Slide 6 — Insight & rencana perbaikan
- ‹Insight 1 → aksi›. Contoh: *istilah teknis membingungkan* → tambah tooltip/glosarium awam.
- ‹Insight 2 → aksi›. Contoh: *input sempit di HP* → sudah dimitigasi dengan expander; pertimbangkan slider.
- ‹Insight 3 → aksi›.

---

### Cara mengisi
1. Ekspor respons Google Form → CSV.
2. `python scripts/analyze_user_testing.py --input <responses.csv>`
3. Salin angka dari `analysis_summary.md` & tema dari `qualitative_responses.md`.
4. Sisipkan PNG dari `reports/figures/user_testing/` ke slide.
