"""
Modul Named Entity Recognition (NER) Sederhana - Rule & Dictionary Based
---------------------------------------------------------------------------
Mendeteksi entitas dalam teks Bahasa Indonesia: PERSON, LOCATION,
ORGANIZATION, DATE, MONEY tanpa menggunakan library NLP eksternal
(seperti spaCy). Pendekatan ini menggabungkan:
  1. Regex untuk pola terstruktur (tanggal, uang)
  2. Kamus kata untuk lokasi & indikator organisasi
  3. Heuristik kapitalisasi untuk mendeteksi nama orang

Ini adalah pendekatan umum untuk bootstrap NER sebelum melatih model
machine learning yang lebih canggih -- mirip dengan cara kerja label
awal ("weak supervision") pada banyak pipeline data annotation.
"""

import re

# ---- Kamus dasar ----
LOKASI = {
    "jakarta", "bandung", "surabaya", "medan", "semarang", "makassar",
    "palembang", "yogyakarta", "jogja", "denpasar", "bali", "indramayu",
    "majalengka", "karawang", "bogor", "bekasi", "tangerang", "depok",
    "indonesia", "malaysia", "singapura", "jawa barat", "jawa tengah",
    "jawa timur", "sumatra", "kalimantan", "sulawesi", "papua",
}

ORG_KEYWORDS = {
    "pt", "cv", "ud", "koperasi", "yayasan", "kementerian", "dinas",
    "universitas", "institut", "sekolah", "smk", "sma", "perusahaan",
}

ORG_KNOWN = {
    "honda prospect motor", "myskill", "google", "microsoft", "amazon",
    "appen", "toloka", "clickworker", "upwork", "fiverr",
}

BULAN = (
    "januari|februari|maret|april|mei|juni|juli|agustus|"
    "september|oktober|november|desember"
)

# ---- Regex patterns ----
DATE_PATTERN = re.compile(
    rf"\b\d{{1,2}}\s+(?:{BULAN})\s+\d{{4}}\b"          # 23 Maret 2025
    rf"|\b(?:{BULAN})\s+\d{{4}}\b"                      # Maret 2025
    rf"|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b",             # 23/03/2025
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"\bRp\s?[\d.,]+(?:\s?(?:juta|miliar|ribu))?\b"
    r"|\$\s?[\d.,]+(?:\s?(?:USD|dollar))?\b",
    re.IGNORECASE,
)


def _find_capitalized_sequences(text):
    """Cari urutan kata berhuruf kapital (kemungkinan nama orang).

    Kata tunggal berhuruf kapital di awal kalimat diabaikan (karena itu cuma
    aturan kapitalisasi kalimat biasa), TAPI kalau kata itu diikuti kata
    kapital lain (misal "Budi Santoso" di awal kalimat), tetap dianggap
    kandidat nama -- karena kombinasi 2+ kata kapital berurutan jarang
    terjadi kecuali memang nama orang.
    """
    words = text.split()
    entities = []
    current = []

    def flush(start_idx):
        if not current:
            return
        phrase = " ".join(current)
        is_sentence_start = start_idx == 0 or words[start_idx - 1].endswith((".", "!", "?"))
        # Buang kalau cuma 1 kata DAN itu di awal kalimat (kemungkinan besar
        # cuma huruf kapital awal kalimat biasa, bukan nama).
        if len(current) == 1 and is_sentence_start:
            return
        if phrase.lower() not in LOKASI and phrase.lower() not in ORG_KNOWN:
            entities.append((phrase, start_idx))

    current_start_idx = None
    for i, word in enumerate(words):
        clean = re.sub(r"[^\w\s]", "", word)
        is_capitalized = clean[:1].isupper() and clean[1:].islower() and len(clean) > 1

        if is_capitalized:
            if not current:
                current_start_idx = i
            current.append(clean)
        else:
            flush(current_start_idx)
            current = []
            current_start_idx = None

    flush(current_start_idx)
    return entities


def extract_entities(text):
    """
    Mengekstrak entitas dari teks. Mengembalikan list of dict:
    {"text": ..., "label": ..., "start": ..., "end": ...}
    """
    entities = []

    # 1. DATE
    for m in DATE_PATTERN.finditer(text):
        entities.append({"text": m.group(), "label": "DATE", "start": m.start(), "end": m.end()})

    # 2. MONEY
    for m in MONEY_PATTERN.finditer(text):
        entities.append({"text": m.group(), "label": "MONEY", "start": m.start(), "end": m.end()})

    # 3. LOCATION (dictionary lookup, case-insensitive, whole word)
    for loc in LOKASI:
        for m in re.finditer(rf"\b{re.escape(loc)}\b", text, re.IGNORECASE):
            entities.append({"text": m.group(), "label": "LOCATION", "start": m.start(), "end": m.end()})

    # 4. ORGANIZATION (known orgs)
    for org in ORG_KNOWN:
        for m in re.finditer(rf"\b{re.escape(org)}\b", text, re.IGNORECASE):
            entities.append({"text": m.group(), "label": "ORGANIZATION", "start": m.start(), "end": m.end()})

    # 5. ORGANIZATION (keyword + following capitalized words, e.g. "PT Honda Prospect Motor")
    for kw in ORG_KEYWORDS:
        # Catatan: keyword dicocokkan case-insensitive, tapi kata setelahnya
        # (nama organisasi) HARUS tetap case-sensitive supaya [A-Z] benar-benar
        # berarti huruf kapital -- kalau seluruh pattern pakai re.IGNORECASE,
        # [A-Z] akan ikut cocok dengan huruf kecil juga (bug umum regex).
        pattern = rf"\b(?i:{re.escape(kw)})\b\s+([A-Z][\w]*(?:\s+[A-Z][\w]*)*)"
        for m in re.finditer(pattern, text):
            entities.append({"text": m.group(), "label": "ORGANIZATION", "start": m.start(), "end": m.end()})

    # 6. PERSON (capitalized sequences not already classified as LOCATION/ORG)
    covered_spans = [(e["start"], e["end"]) for e in entities]
    for phrase, idx in _find_capitalized_sequences(text):
        pos = text.find(phrase)
        if pos == -1:
            continue
        end = pos + len(phrase)
        overlap = any(not (end <= s or pos >= e) for s, e in covered_spans)
        if not overlap:
            entities.append({"text": phrase, "label": "PERSON", "start": pos, "end": end})

    # Urutkan berdasarkan posisi, lalu selesaikan tumpang tindih span:
    # kalau dua entitas saling overlap, ambil yang rentangnya lebih panjang
    # (contoh: "PT Honda Prospect Motor" menang atas "Honda Prospect Motor").
    seen = set()
    deduped = []
    for e in sorted(entities, key=lambda x: x["start"]):
        key = (e["start"], e["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    deduped.sort(key=lambda x: (x["end"] - x["start"]), reverse=True)
    final_entities = []
    occupied = []
    for e in deduped:
        overlap = any(not (e["end"] <= s or e["start"] >= en) for s, en in occupied)
        if not overlap:
            final_entities.append(e)
            occupied.append((e["start"], e["end"]))

    final_entities.sort(key=lambda x: x["start"])
    return final_entities


if __name__ == "__main__":
    contoh = [
        "Sutrisno bekerja di PT Honda Prospect Motor sejak April 2021 di Karawang.",
        "Rapat akan diadakan di Jakarta pada 15 Oktober 2026 dengan anggaran Rp 50 juta.",
        "Budi Santoso mendapatkan sertifikat dari MySkill pada 23 Maret 2025.",
    ]
    for teks in contoh:
        print(f"\nTeks: {teks}")
        for ent in extract_entities(teks):
            print(f"  - {ent['text']!r} -> {ent['label']}")
