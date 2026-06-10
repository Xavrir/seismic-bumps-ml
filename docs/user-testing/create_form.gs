/**
 * Auto-builds the "Pengujian Pengguna — Seismic Risk Console" Google Form.
 *
 * HOW TO USE
 *   1. Go to https://script.google.com → New project.
 *   2. Paste this whole file, replacing the default Code.gs contents.
 *   3. Run `createSeismicUserTestingForm` (authorize when prompted — first run only).
 *   4. Open View → Logs (or Executions) to get the form's Edit URL and Live URL.
 *   5. Share the Live URL with participants. Export responses via the form's
 *      Responses tab → "..." → Download responses (.csv), then run
 *      scripts/analyze_user_testing.py.
 *
 * The question titles keep the [code] prefix so the exported CSV headers carry the
 * code and scripts/analyze_user_testing.py matches columns automatically.
 */

function createSeismicUserTestingForm() {
  var form = FormApp.create('Pengujian Pengguna — Seismic Risk Console');

  form.setDescription(
    'Terima kasih sudah membantu. Ini DEMO RISET, bukan perangkat keselamatan tambang ' +
    'resmi. Kami menguji APLIKASINYA, bukan menguji Anda — tidak ada jawaban benar/salah. ' +
    'Form ini ANONIM (tanpa nama/email). Buka aplikasi di ' +
    'https://coalmine-seismic-risk.streamlit.app/ lalu kerjakan 5 tugas yang dipandu ' +
    'fasilitator sebelum mengisi.'
  );
  form.setProgressBar(true);
  form.setShowLinkToRespondAgain(false);
  try { form.setCollectEmail(false); } catch (e) { /* older runtimes: toggle manually */ }

  // ---- Bagian A — Latar belakang (anonim) --------------------------------
  form.addPageBreakItem().setTitle('Bagian A — Latar belakang (anonim)');
  form.addTextItem()
      .setTitle('[pid] Kode peserta (diisi fasilitator, mis. P1)')
      .setRequired(true);
  form.addTextItem()
      .setTitle('[major] Program studi / jurusan')
      .setRequired(true);
  form.addMultipleChoiceItem()
      .setTitle('[year] Tahun / jenjang kuliah')
      .setChoiceValues(['Tahun 1', 'Tahun 2', 'Tahun 3', 'Tahun 4', 'Pascasarjana'])
      .setRequired(true);
  addScale_(form, '[ml_fam] Seberapa familiar kamu dengan machine learning?',
            'Tidak familiar', 'Sangat familiar');
  addScale_(form, '[dash_fam] Seberapa sering kamu memakai aplikasi data / dashboard?',
            'Tidak pernah', 'Sangat sering');
  form.addMultipleChoiceItem()
      .setTitle('[device] Perangkat yang dipakai untuk tes ini')
      .setChoiceValues(['HP', 'Laptop atau Desktop', 'Tablet'])
      .setRequired(true);

  // ---- Bagian B — Tugas & tingkat kesulitan ------------------------------
  form.addPageBreakItem()
      .setTitle('Bagian B — Tugas & tingkat kesulitan')
      .setHelpText('Kerjakan tiap tugas di aplikasi, lalu isi status dan tingkat kesulitan. ' +
                   'Kesulitan: 1 = Sangat mudah, 5 = Sangat sulit.');
  var tasks = [
    ['t1', 'T1 — Memahami fungsi aplikasi dari panel "Start here"'],
    ['t2', 'T2 — Membaca risk gauge, risk level, dan verdict di "Try one shift"'],
    ['t3', 'T3 — Mengubah satu input lalu mengamati perubahan risiko'],
    ['t4', 'T4 — Upload CSV dan menemukan jumlah shift "dangerous"'],
    ['t5', 'T5 — Menemukan recall model & arti threshold']
  ];
  tasks.forEach(function (t) {
    form.addMultipleChoiceItem()
        .setTitle('[' + t[0] + '_status] ' + t[1] + ': status')
        .setChoiceValues(['Selesai', 'Sebagian', 'Gagal'])
        .setRequired(true);
    addScale_(form, '[' + t[0] + '_diff] ' + t[1] + ': tingkat kesulitan',
              'Sangat mudah', 'Sangat sulit');
  });

  // ---- Bagian C — Usability (SUS, 10 item) -------------------------------
  form.addPageBreakItem()
      .setTitle('Bagian C — Usability (SUS)')
      .setHelpText('Skala 1–5. 1 = Sangat tidak setuju, 5 = Sangat setuju.');
  var sus = [
    ['sus1',  'Saya rasa saya ingin sering menggunakan aplikasi ini.'],
    ['sus2',  'Saya merasa aplikasi ini terlalu rumit.'],
    ['sus3',  'Saya rasa aplikasi ini mudah digunakan.'],
    ['sus4',  'Saya rasa saya butuh bantuan orang teknis untuk bisa memakai aplikasi ini.'],
    ['sus5',  'Saya rasa fitur-fitur aplikasi ini terintegrasi dengan baik.'],
    ['sus6',  'Saya rasa ada terlalu banyak ketidak-konsistenan dalam aplikasi ini.'],
    ['sus7',  'Saya rasa kebanyakan orang akan cepat belajar memakai aplikasi ini.'],
    ['sus8',  'Saya merasa aplikasi ini sangat janggal/membingungkan saat digunakan.'],
    ['sus9',  'Saya merasa percaya diri saat menggunakan aplikasi ini.'],
    ['sus10', 'Saya perlu belajar banyak hal dulu sebelum bisa memakai aplikasi ini.']
  ];
  sus.forEach(function (it) {
    addScale_(form, '[' + it[0] + '] ' + it[1], 'Sangat tidak setuju', 'Sangat setuju');
  });

  // ---- Bagian D — Usefulness / kegunaan ----------------------------------
  form.addPageBreakItem()
      .setTitle('Bagian D — Usefulness / kegunaan')
      .setHelpText('Skala 1–5. 1 = Sangat tidak setuju, 5 = Sangat setuju.');
  var use = [
    ['use1', 'Saya paham arti risk level (low / watch / dangerous) yang ditampilkan.'],
    ['use2', 'Output aplikasi (skor & level risiko) terasa berguna untuk menilai bahaya shift.'],
    ['use3', 'Saya percaya hasil aplikasi ini layak dijadikan alat bantu pengambilan keputusan.'],
    ['use4', 'Tab Model evidence / Methodology membantu saya memahami cara kerja & keterbatasan model.'],
    ['use5', 'Secara keseluruhan, aplikasi ini menjelaskan prediksinya dengan transparan.']
  ];
  use.forEach(function (it) {
    addScale_(form, '[' + it[0] + '] ' + it[1], 'Sangat tidak setuju', 'Sangat setuju');
  });

  // ---- Bagian E — Umpan balik terbuka ------------------------------------
  form.addPageBreakItem().setTitle('Bagian E — Umpan balik terbuka');
  form.addParagraphTextItem()
      .setTitle('[q_confuse] Bagian mana yang paling membingungkan atau sulit? Kenapa?');
  form.addParagraphTextItem()
      .setTitle('[q_like] Apa yang kamu sukai dari aplikasi ini?');
  form.addParagraphTextItem()
      .setTitle('[q_suggest] Saran perbaikan apa yang kamu punya?');
  form.addParagraphTextItem()
      .setTitle('[q_clarity] Menurutmu, apakah output risiko sudah jelas/mudah dipahami? Jelaskan.');

  Logger.log('FORM CREATED');
  Logger.log('Edit URL : ' + form.getEditUrl());
  Logger.log('Live URL : ' + form.getPublishedUrl());
}

/** Add a 1–5 linear-scale, required, with low/high labels. */
function addScale_(form, title, lowLabel, highLabel) {
  form.addScaleItem()
      .setTitle(title)
      .setBounds(1, 5)
      .setLabels(lowLabel, highLabel)
      .setRequired(true);
}
