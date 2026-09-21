# 🗂️ Multi-Format Annotation Dashboard

Platform anotasi data yang menggabungkan dua jenis pekerjaan data annotation paling umum di industri AI/ML: **Text NER (Named Entity Recognition)** dan **Image Bounding Box Annotation** — dalam satu dashboard, mirip konsep tools industri seperti Labelbox atau CVAT.

Project ini dibuat sebagai portofolio lanjutan, melengkapi project sebelumnya (sentiment classification) dengan kemampuan anotasi yang lebih beragam — sesuai dengan jenis pekerjaan yang paling sering diminta di platform freelance data annotation (Upwork, OpenTrain, dll).

## Fitur

### 📝 Text NER (Named Entity Recognition)
- Mendeteksi entitas **PERSON**, **LOCATION**, **ORGANIZATION**, **DATE**, dan **MONEY** dari teks Bahasa Indonesia
- Highlight visual entitas langsung pada teks
- Dibangun dari nol menggunakan pendekatan rule & dictionary-based (regex + kamus kata), tanpa library NLP eksternal seperti spaCy
- Bisa pakai teks contoh atau masukkan teks sendiri

### 🖼️ Image Annotation (Bounding Box)
- Gambar kotak (bounding box) pada gambar dengan mengatur koordinat
- Beri label pada tiap box (buah, kotak/paket, rambu, lainnya)
- Hitung **IoU (Intersection over Union)** — metrik standar industri untuk menilai akurasi bounding box
- Export hasil anotasi sebagai CSV

## Tech Stack

- **Python 3**
- **Pandas** — pengolahan data
- **Pillow (PIL)** — pemrosesan dan menggambar pada gambar
- **Streamlit** — antarmuka web interaktif

## Struktur Project

```
annotation_dashboard/
├── sample_images/          # Gambar contoh (synthetic) untuk image annotation
│   ├── scene_fruits.png
│   ├── scene_signs.png
│   └── scene_boxes.png
├── sample_texts.csv        # Dataset contoh teks untuk NER
├── ner_module.py            # Modul NER rule & dictionary-based
├── image_module.py          # Modul bounding box drawing + IoU calculation
├── app.py                   # Aplikasi utama Streamlit
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```

3. (Opsional) Test modul NER secara terpisah:
   ```
   python ner_module.py
   ```

## Cara Kerja NER (Rule & Dictionary-Based)

1. **Pattern matching (regex)** untuk entitas terstruktur: tanggal (contoh: "23 Maret 2025") dan nominal uang (contoh: "Rp 50 juta")
2. **Dictionary lookup** untuk lokasi (kota/provinsi Indonesia) dan organisasi yang dikenal
3. **Keyword + capitalization** untuk mendeteksi organisasi (contoh: "PT" diikuti kata berhuruf kapital → "PT Honda Prospect Motor")
4. **Heuristik kapitalisasi** untuk mendeteksi nama orang (urutan kata berhuruf kapital yang bukan lokasi/organisasi)
5. **Resolusi overlap** — kalau ada beberapa entitas yang tumpang tindih posisinya, sistem otomatis pilih yang rentang teksnya paling lengkap

### Keterbatasan yang Disadari

- Nama orang yang terdiri dari **1 kata** dan berada di **awal kalimat** (contoh: "Sutrisno bekerja...") tidak terdeteksi — sistem sengaja mengabaikan kata tunggal di awal kalimat untuk menghindari false positive dari aturan kapitalisasi kalimat biasa
- Kombinasi 2 kata berhuruf kapital yang bukan nama (contoh: "Data Analyst") kadang salah terdeteksi sebagai PERSON — ini trade-off dari pendekatan rule-based tanpa training data
- Pendekatan ini adalah **baseline/bootstrap**, bukan pengganti model NER terlatih (seperti spaCy atau transformer-based NER) — tapi cukup untuk mendemonstrasikan konsep dan alur kerja anotasi

## Cara Kerja Image Annotation

Bounding box digambar berdasarkan input koordinat manual (x, y, lebar, tinggi) — bukan drag-and-drop, karena keterbatasan library yang tersedia saat pengembangan. Namun konsep dan output-nya tetap merepresentasikan proses anotasi bounding box yang sesungguhnya, termasuk perhitungan IoU yang dipakai secara nyata di industri untuk mengukur kualitas anotasi.

**IoU (Intersection over Union)** dihitung dengan membagi luas area irisan (intersection) dua box dengan luas area gabungan (union) keduanya. Skor mendekati 1.0 berarti dua box hampir identik; skor 0 berarti tidak ada tumpang tindih sama sekali.

## Pengembangan Selanjutnya (Ide)

- Ganti input koordinat manual dengan drag-and-drop canvas (butuh library `streamlit-drawable-canvas`)
- Tambahkan model NER berbasis machine learning (spaCy/transformers) sebagai perbandingan dengan versi rule-based
- Dukung upload gambar sendiri (bukan cuma gambar contoh)
- Tambah lebih banyak label kategori dan kamus entitas

---

*Project ini dibuat sebagai bagian dari portofolio pembelajaran Software Engineering dan eksplorasi bidang Data Annotation/AI.*
