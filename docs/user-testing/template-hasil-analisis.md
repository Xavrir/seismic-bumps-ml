# Hasil Pengujian Pengguna — Seismic Risk Console

Terisi dari 12 respons nyata (anonim, P1–P12). Angka & grafik dari
`analysis_summary.md`, `qualitative_responses.md`, dan `reports/figures/user_testing/*.png`
(`scripts/analyze_user_testing.py`). Siap dipindah ke slide PPT bagian *User Testing*.

> Memenuhi syarat wajib: **12 pengguna nyata** di luar tim, latar belakang anonim + skenario
> penggunaan, umpan balik **usability + usefulness** (kuantitatif + kualitatif), dianalisis.

---

## Slide 1 — Metodologi
- **Peserta:** 12 mahasiswa, bukan anggota tim, anonim (P1–P12). Tanpa nama/NIM.
- **Metode:** *task-based usability test* pada aplikasi ter-deploy
  (https://coalmine-seismic-risk.streamlit.app/) — bukan role-play.
- **Skenario:** tiap peserta menyelesaikan 5 tugas nyata: (T1) memahami fungsi dari panel
  "Start here", (T2) membaca risk gauge/level/verdict, (T3) mengubah input & mengamati
  perubahan risiko, (T4) upload CSV & menemukan jumlah "dangerous", (T5) menemukan recall &
  arti threshold.
- **Instrumen:** Google Form — latar belakang, status & kesulitan per tugas, **SUS** (10 item),
  **usefulness** (5 item), 4 pertanyaan terbuka.
- **Analisis:** `scripts/analyze_user_testing.py` (SUS dengan reverse-scoring item negatif,
  rata-rata kegunaan, kesulitan & keberhasilan tugas, tema kualitatif).

## Slide 2 — Latar belakang peserta (anonim)
Sebaran: **ML familiarity rata-rata 4.2/5**, **dashboard 4.2/5**; perangkat **8 Laptop/Desktop,
2 Tablet, 2 HP**. (Catatan: sampel cenderung melek teknis/CS — lihat keterbatasan.)

Grafik: `background_summary.png`.

| Peserta | Prodi | ML (1–5) | Dashboard (1–5) | Perangkat |
|---|---|---|---|---|
| P1 | Computer Science | 3 | 4 | Laptop/Desktop |
| P2 | Computer Science | 4 | 5 | Laptop/Desktop |
| P3 | Teknik Informatika | 4 | 4 | Tablet |
| P4 | Computer Science | 5 | 4 | Laptop/Desktop |
| P5 | Computer Science | 5 | 5 | Laptop/Desktop |
| P6 | Cyber Security | 4 | 3 | HP |
| P7 | Computer Science | 3 | 4 | Laptop/Desktop |
| P8 | Computer Science | 4 | 5 | Laptop/Desktop |
| P9 | Data Science | 4 | 4 | Tablet |
| P10 | Computer Science | 5 | 4 | Laptop/Desktop |
| P11 | Computer Science | 5 | 5 | Laptop/Desktop |
| P12 | Computer Science | 4 | 3 | HP |

## Slide 3 — Hasil kuantitatif: Usability (SUS)
- **Skor SUS rata-rata: 87.5 / 100 → grade A (Excellent)** (rujukan industri ≈ 68).
- Sebaran per peserta: 75 atau 100 (mayoritas tinggi). Grafik: `sus_per_participant.png`.
- **Interpretasi:** usability dipersepsikan sangat baik — jauh di atas rata-rata; aplikasi
  dirasa mudah dipakai bahkan oleh pengguna pertama kali.

## Slide 4 — Hasil kuantitatif: Usefulness & tugas
**Usefulness (1–5): rata-rata 4.33 (87%).** Grafik: `usefulness_items.png`.

| Item kegunaan | Rata-rata |
|---|---|
| Paham arti risk level | 4.50 |
| Output berguna menilai bahaya | 4.50 |
| Penjelasan transparan | 4.50 |
| Percaya sebagai alat bantu | 4.33 |
| Tab Evidence/Methodology membantu | **3.83** (terendah) |

**Tugas — kesulitan & keberhasilan.** Grafik: `task_metrics.png`.

| Tugas | Kesulitan (1–5) | Keberhasilan |
|---|---|---|
| T1 Memahami fungsi | 1.5 | 100% |
| T2 Baca gauge/level/verdict | 1.7 | 100% |
| T3 Ubah input | 2.0 | 92% |
| **T4 Upload CSV** | **2.8 (tersulit)** | **75% (terendah)** |
| T5 Recall & threshold | 1.8 | 92% |

**Temuan utama:** memahami & membaca output (T1, T2) sangat mudah (100% sukses); **alur Upload
CSV (T4) adalah titik tersulit** (kesulitan 2.8, sukses 75%), dan item kegunaan terendah adalah
pemahaman tab **Methodology/Evidence** (3.83) — konsisten dengan keluhan istilah teknis.

## Slide 5 — Temuan kualitatif (tema)
Dari pertanyaan terbuka (6 peserta memberi komentar):
- **Disukai:** *risk gauge* + *verdict* langsung terbaca; kategori **low/watch/dangerous** &
  warna mudah dipahami; tab **Model evidence/Methodology** membuat prediksi terasa transparan;
  alur *single shift → batch CSV* terasa lengkap.
- **Kendala:** (1) **Upload CSV** perlu contoh format yang jelas; (2) istilah **recall &
  threshold** perlu penjelasan singkat; (3) perubahan input → risiko perlu **penanda visual**;
  (4) jumlah **"dangerous"** pada hasil batch perlu di-*highlight*; (5) tampilan **mobile padat**.
- **Saran:** contoh CSV yang bisa diunduh langsung; tooltip istilah teknis; highlight saat input
  menaikkan/menurunkan risiko; highlight otomatis baris/ringkasan dangerous; optimasi spasi & teks
  untuk mobile.

## Slide 6 — Insight & rencana perbaikan (sudah ditindaklanjuti)
Umpan balik langsung kami terjemahkan jadi perbaikan aplikasi:

| Temuan pengguna | Tindakan | Status |
|---|---|---|
| Istilah recall/threshold membingungkan | Tooltip pada metrik, glosarium "What do these terms mean?", hover pada readout verdict | ✅ Selesai |
| Upload CSV kurang jelas | Contoh CSV ditampilkan inline + label langkah "Download example CSV (template)" | ✅ Selesai |
| Jumlah "dangerous" kurang menonjol | Banner peringatan + kartu *dangerous* di-*highlight* saat >0 | ✅ Selesai |
| Tampilan mobile padat | Spasi & ukuran teks dirapatkan untuk layar < 820px | ✅ Selesai |
| Perubahan input → risiko perlu penanda | Indikator naik/turun risiko saat input diubah | ⏳ Rencana |

**Narasi PPT:** kami menguji → menemukan titik gesek (Upload CSV, istilah teknis, mobile) →
memperbaikinya di iterasi yang sama. Siklus tes-perbaiki ini adalah inti dari user testing.

## Keterbatasan (sebutkan di PPT)
- **Sampel kecil (n=12)** — laporkan angka apa adanya, hindari generalisasi berlebihan.
- **Sampel cenderung teknis** (mayoritas Computer Science, ML familiarity 4.2/5) — usability bagi
  pengguna awam non-teknis belum banyak teruji.
- Hanya **6 dari 12** mengisi pertanyaan terbuka — tema kualitatif dari subset tersebut.

---

### Cara memperbarui (jika menambah respons)
1. Ekspor Google Form → CSV → buang kolom identitas (NIM/Tahun).
2. `python scripts/analyze_user_testing.py --input <responses_anon.csv>`
3. Salin angka dari `analysis_summary.md` & tema dari `qualitative_responses.md`; sisipkan PNG
   dari `reports/figures/user_testing/`.
