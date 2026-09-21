"""
Multi-Format Annotation Dashboard
------------------------------------
Platform anotasi data yang menggabungkan dua modul:
1. Text NER (Named Entity Recognition) - deteksi entitas dalam teks
2. Image Annotation - bounding box labeling pada gambar

Cara jalankan:
    streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st
from ner_module import extract_entities
from image_module import draw_boxes, compute_iou, BOX_COLORS

st.set_page_config(page_title="Multi-Format Annotation Dashboard", layout="wide")

st.title("🗂️ Multi-Format Annotation Dashboard")
st.caption(
    "Platform anotasi data mensimulasikan tools seperti Labelbox/CVAT — "
    "mendukung anotasi teks (NER) dan gambar (bounding box) dalam satu tempat."
)

tab_ner, tab_image, tab_about = st.tabs(["📝 Text NER", "🖼️ Image Annotation", "ℹ️ Tentang"])

# =====================================================================
# TAB 1: TEXT NER
# =====================================================================
with tab_ner:
    st.subheader("Named Entity Recognition (NER)")
    st.write(
        "Sistem mendeteksi entitas **PERSON**, **LOCATION**, **ORGANIZATION**, "
        "**DATE**, dan **MONEY** dari teks menggunakan pendekatan rule & dictionary-based."
    )

    LABEL_COLORS = {
        "PERSON": "#FF6B6B",
        "LOCATION": "#4ECDC4",
        "ORGANIZATION": "#FFD166",
        "DATE": "#6A8EAE",
        "MONEY": "#95D5B2",
    }

    @st.cache_data
    def load_texts():
        return pd.read_csv("sample_texts.csv")

    df_texts = load_texts()

    col_select, col_custom = st.columns([2, 1])
    with col_select:
        selected_id = st.selectbox(
            "Pilih teks contoh",
            df_texts["id"].tolist(),
            format_func=lambda x: f"Teks #{x}: {df_texts[df_texts['id']==x]['text'].values[0][:50]}...",
        )
    text_input = df_texts[df_texts["id"] == selected_id]["text"].values[0]

    custom_text = st.text_area("Atau masukkan teks kamu sendiri:", value=text_input, height=100)

    entities = extract_entities(custom_text)

    # Render teks dengan highlight
    st.markdown("**Hasil Anotasi:**")
    highlighted = ""
    last_end = 0
    for ent in entities:
        highlighted += custom_text[last_end:ent["start"]]
        color = LABEL_COLORS.get(ent["label"], "#CCCCCC")
        highlighted += (
            f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px;">'
            f'{ent["text"]} <b style="font-size:0.7em;">[{ent["label"]}]</b></span>'
        )
        last_end = ent["end"]
    highlighted += custom_text[last_end:]

    st.markdown(f'<div style="line-height:2; font-size:1.1em;">{highlighted}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Daftar Entitas Terdeteksi:**")
    if entities:
        df_entities = pd.DataFrame(entities)[["text", "label", "start", "end"]]
        df_entities.columns = ["Teks", "Label", "Posisi Awal", "Posisi Akhir"]
        st.dataframe(df_entities, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada entitas terdeteksi pada teks ini.")

    # Legend
    st.markdown("**Legenda Label:**")
    legend_cols = st.columns(len(LABEL_COLORS))
    for col, (label, color) in zip(legend_cols, LABEL_COLORS.items()):
        col.markdown(
            f'<span style="background-color:{color}; padding:2px 8px; border-radius:4px;">{label}</span>',
            unsafe_allow_html=True,
        )

# =====================================================================
# TAB 2: IMAGE ANNOTATION
# =====================================================================
with tab_image:
    st.subheader("Image Bounding Box Annotation")
    st.write(
        "Gambar kotak (bounding box) di sekitar objek dengan mengatur koordinat, "
        "lalu beri label. Cocok untuk simulasi anotasi computer vision."
    )

    image_dir = "sample_images"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    selected_image = st.selectbox("Pilih gambar", image_files)
    image_path = os.path.join(image_dir, selected_image)

    if "boxes" not in st.session_state:
        st.session_state.boxes = []
    if "last_image" not in st.session_state or st.session_state.last_image != selected_image:
        st.session_state.boxes = []
        st.session_state.last_image = selected_image

    col_controls, col_preview = st.columns([1, 2])

    with col_controls:
        st.markdown("**Tambah Bounding Box**")
        x = st.slider("Posisi X", 0, 600, 50)
        y = st.slider("Posisi Y", 0, 400, 50)
        w = st.slider("Lebar", 10, 400, 100)
        h = st.slider("Tinggi", 10, 400, 100)
        label = st.selectbox("Label objek", list(BOX_COLORS.keys()))

        if st.button("➕ Tambah Box", use_container_width=True):
            st.session_state.boxes.append({"x": x, "y": y, "w": w, "h": h, "label": label})

        if st.button("🗑️ Hapus Semua Box", use_container_width=True):
            st.session_state.boxes = []

        if st.session_state.boxes:
            st.markdown("**Box yang sudah ditambahkan:**")
            for i, box in enumerate(st.session_state.boxes):
                st.write(f"{i+1}. {box['label']} — x:{box['x']}, y:{box['y']}, w:{box['w']}, h:{box['h']}")

    with col_preview:
        result_img = draw_boxes(image_path, st.session_state.boxes)
        st.image(result_img, use_container_width=True, caption=f"Preview: {selected_image}")

    st.divider()
    st.markdown("**🎯 Evaluasi Akurasi (IoU Score)**")
    st.write(
        "IoU (Intersection over Union) mengukur seberapa akurat bounding box kamu "
        "dibanding box referensi. Skor 1.0 = sempurna, di atas 0.5 umumnya dianggap baik."
    )
    if len(st.session_state.boxes) >= 2:
        box_a = st.session_state.boxes[0]
        box_b = st.session_state.boxes[1]
        iou_score = compute_iou(box_a, box_b)
        st.metric("IoU antara Box #1 dan Box #2", f"{iou_score:.3f}")
    else:
        st.info("Tambahkan minimal 2 box untuk menghitung IoU antar box.")

    # Export
    if st.session_state.boxes:
        df_boxes = pd.DataFrame(st.session_state.boxes)
        csv = df_boxes.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Anotasi sebagai CSV", csv, "image_annotations.csv", "text/csv")

# =====================================================================
# TAB 3: ABOUT
# =====================================================================
with tab_about:
    st.subheader("Tentang Project Ini")
    st.markdown("""
    Project ini dibangun untuk mendemonstrasikan pemahaman terhadap dua jenis
    pekerjaan data annotation yang paling umum di industri AI/ML:

    **1. Text Annotation (NER)**
    - Named Entity Recognition mengenali entitas seperti nama orang, lokasi, organisasi
    - Digunakan untuk melatih model NLP (chatbot, ekstraksi informasi, dll)
    - Pendekatan di sini: rule & dictionary-based (dibangun dari nol tanpa library eksternal)

    **2. Image Annotation (Bounding Box)**
    - Digunakan untuk melatih model computer vision (deteksi objek, self-driving car, dll)
    - IoU (Intersection over Union) adalah metrik standar industri untuk menilai
      seberapa akurat sebuah bounding box dibanding ground truth

    **Keterbatasan yang disadari:**
    - NER berbasis rule/dictionary punya keterbatasan dibanding model ML terlatih —
      misalnya nama tunggal di awal kalimat kadang tidak terdeteksi
    - Image annotation di sini pakai input koordinat manual (bukan drag-and-drop),
      karena keterbatasan library yang tersedia — namun konsep dan output-nya
      tetap merepresentasikan proses anotasi bounding box yang sesungguhnya

    **Tech stack:** Python, Pandas, Pillow (PIL), Streamlit
    """)
