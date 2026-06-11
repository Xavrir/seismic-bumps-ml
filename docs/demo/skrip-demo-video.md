# Skrip Demo Video — Seismic Risk Console

Video demonstrasi aplikasi untuk submission (bobot 10%: *deployment demonstration* + *explanation
clarity*). Narasi **voice-over Bahasa Indonesia**, demo pakai **app yang sudah di-deploy** di
https://coalmine-seismic-risk.streamlit.app/.

- **Target durasi:** ~1 menit 40 (sekitar 95–110 detik). Tugas tidak membatasi durasi; intro dengan
  justifikasi model + bagian methodology yang lengkap masing-masing butuh waktu. Kalau memang harus
  pendek, pakai *Intro versi pendek* dan *Penutup versi pendek* di bawah.
- **File demo CSV:** `docs/demo/seismic_demo_shifts.csv` (9 baris: 5 low, 2 watch, 2 dangerous).
  Yang dangerous diambil dari shift yang benar-benar berbahaya di data asli, jadi banner merahnya
  jujur, bukan rekayasa.
- Sebelum rekaman: buka app + simpan `seismic_demo_shifts.csv` di desktop biar pas demo nggak nyari
  file.

---

## Naskah voice-over (baca santai, jeda di tiap titik)

**[0:00–0:24 · Header + justifikasi model]**
> "Hai. Ini Seismic Risk Console, aplikasi buat nilai seberapa bahaya satu shift kerja di tambang
> batu bara. Di balik layar, kita udah banding beberapa model, dan yang dipilih Logistic Regression.
> Alasannya, recall-nya paling tinggi dan probabilitasnya terkalibrasi, jadi angkanya bisa dipercaya.
> Di kasus tambang, kelewatan satu bahaya itu jauh lebih mahal daripada alarm palsu, kira-kira
> sepuluh banding satu. Makanya threshold-nya kita pasang rendah, di nol koma nol delapan, biar
> modelnya cenderung cari-aman. Modelnya kita kunci, jadi hasilnya selalu konsisten."

**[0:24–0:42 · Tab "Try one shift"]**
> "Kita coba satu shift dulu. Aku ambil contoh data yang udah disiapin, buka panel input, terus ubah
> salah satu nilainya. Lihat, gauge-nya langsung gerak, skornya berubah, dan levelnya kebaca: low,
> watch, atau dangerous. Di bawahnya ada probabilitas sama threshold-nya, jadi jelas kenapa dia
> ngasih label itu."

**[0:42–1:02 · Tab "Upload CSV"]**

> "Kalau datanya banyak, tinggal ke tab Upload CSV. Aku download dulu template-nya biar kolomnya pas,
> terus upload file shift-nya. Hasilnya langsung diringkas: berapa yang low, watch, sama dangerous.
> Yang dangerous dikasih banner merah biar nggak kelewat, dan hasilnya bisa di-download lagi."

**[1:02–1:24 · Tab "Model evidence"]**
> "Sekarang bagian yang paling penting buat transparansi: tab Model evidence. Di sini semua angkanya
> kebuka apa adanya, nggak ditutup-tutupin. Recall-nya sekitar nol koma lima delapan. Artinya, dari
> semua shift yang benar-benar bahaya, segitu yang ketangkap modelnya. AUC-nya sekitar nol koma tujuh
> empat, jadi modelnya cukup baik mbedain mana yang bahaya dan mana yang aman. Yang penting,
> probabilitasnya udah dikalibrasi pakai metode isotonic. Jadi kalau modelnya bilang sepuluh persen,
> ya kira-kira beneran sepuluh dari seratus shift kayak gitu yang bahaya, bukan angka asal. Tiap
> metrik juga aku kasih selang kepercayaan sembilan puluh lima persen, biar kelihatan seberapa pasti
> hasilnya."

**[1:24–1:44 · Tab "Methodology"]**

> "Di tab Methodology, dijelasin dasar tiap keputusannya. Kenapa threshold-nya nol koma nol delapan?
> Itu bukan angka asal, tapi dihitung dari matriks biaya sepuluh banding satu tadi: kelewatan satu
> bahaya jauh lebih mahal daripada salah alarm, jadi modelnya sengaja disetel lebih sensitif. Modelnya
> juga dipilih lewat cross-validation, baru dicek sekali di data lockbox biar nggak overfit. Dan kita
> jujur soal batasnya. Performa segini udah mentok di langit-langit dataset ini, dan model yang lebih
> berat malah cuma curang naikin akurasi tanpa nangkep bahaya beneran. Jadi posisinya ini alat bantu
> sama proyek portofolio, bukan buat keselamatan tambang sungguhan. Segitu aja, makasih udah nonton."

### Intro versi pendek (kalau harus ≤60 detik)
> "Hai. Ini Seismic Risk Console, buat nilai bahaya satu shift tambang batu bara pakai model Logistic
> Regression yang kita kunci, disetel cari-aman biar jarang ada bahaya yang kelewat. Yuk lihat cara
> kerjanya."

### Penutup versi pendek (kalau harus ≤60 detik)
> "Terakhir, di tab Model evidence sama Methodology ada angkanya apa adanya, kayak recall sama AUC,
> probabilitas yang udah dikalibrasi, plus alasan kenapa threshold-nya dipasang segini. Jadi
> transparan, bukan kotak hitam. Segitu aja, makasih."

---

## Cue aksi per scene (gerakan layar, samain sama narasi)

Label di bawah ini persis seperti yang muncul di app.

| Waktu | Tab / layar | Aksi di layar |
|-------|-------------|---------------|
| 0:00–0:24 | Header (+ justifikasi) | Buka URL live. Tunjuk judul **"Seismic Risk Console"**, eyebrow "Frozen policy console", dan readout di pojok kanan: model `logreg`, threshold `0.080`, watch floor `0.04`. Pas ngomongin metrik, boleh hover sebentar tab **Model evidence** biar ada yang dilihat, atau tahan di header aja. |
| 0:24–0:42 | **Try one shift** | Tunjuk contoh baris yang sudah ke-load. Buka expander **"Adjust monitoring inputs"**, naikkan satu nilai (mis. `genergy`). Tunjuk **gauge bundar** + badge level yang berubah, lalu baris telemetri (`probability`, `alert threshold`, `dangerous flag`). |
| 0:42–1:02 | **Upload CSV** | Klik **"Download example CSV (template)"** (1 dtk). Lalu lewat **"Upload shift feature CSV"** upload `seismic_demo_shifts.csv`. Tunjuk 3 tile **Low / Watch / Dangerous**, **banner merah**, dan allocation bar; klik **"Download scored CSV"**. |
| 1:02–1:24 | **Model evidence** | Tunjuk tile metrik **Lockbox recall `0.577`**, **Lockbox F2**, **Lockbox AUC `0.735`**. Lalu scroll ke **reliability diagram** (chart kalibrasi tema gelap) + tabel **selang kepercayaan 95%** waktu ngomongin kalibrasi & CI. |
| 1:24–1:44 | **Methodology** | Tunjuk penjelasan threshold (matriks biaya 10:1), bagian pemilihan model lewat CV + cek lockbox, dan paragraf batasan/ceiling. Selesai. |

Yang dilihat saat upload `seismic_demo_shifts.csv`: **Low = 5, Watch = 2, Dangerous = 2**, dan banner
"2 shifts flagged dangerous — review before the shift starts."

---

## Cara rekam

1. Tool: **OBS Studio** atau perekam layar bawaan OS/browser. Resolusi 1080p, rekam satu jendela
   browser (sembunyikan bookmark/tab biar bersih).
2. Latihan diam 1–2 kali dulu biar klik-kliknya pas sama waktu (~1 menit 40), baru rekam voice-over
   sekali jalan (mic bawaan laptop cukup, cari ruangan sepi).
3. Pre-load app + `seismic_demo_shifts.csv` di desktop sebelum mulai, biar nggak ada jeda nunggu /
   nyari file.
4. Trim bagian hening di editor apa pun (Clipchamp / CapCut / DaVinci). Export **MP4**.
5. Upload (Google Drive atau YouTube unlisted), tempel link-nya di Teams.

## Catatan demo CSV

`seismic_demo_shifts.csv` dibuat oleh `scripts/make_demo_csv.py`: ambil data seismic-bumps asli,
skor pakai model frozen, lalu pilih campuran low/watch/dangerous. Model tidak dilatih ulang dan
tidak diubah. Kalau perlu regenerasi: `PYTHONPATH=. python scripts/make_demo_csv.py`.
