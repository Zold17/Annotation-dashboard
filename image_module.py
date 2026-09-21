"""
Modul Image Annotation - Bounding Box Drawing
------------------------------------------------
Menggambar kotak (bounding box) pada gambar berdasarkan koordinat yang
diberikan, dan menghitung IoU (Intersection over Union) untuk membandingkan
akurasi bounding box manual vs referensi -- metrik standar yang dipakai
untuk evaluasi kualitas image annotation di industri computer vision.
"""

from PIL import Image, ImageDraw, ImageFont

BOX_COLORS = {
    "buah": (220, 50, 50),
    "kotak/paket": (60, 120, 220),
    "rambu": (240, 140, 20),
    "lainnya": (140, 60, 200),
}


def draw_boxes(image_path, boxes):
    """
    Menggambar bounding box pada gambar.
    boxes: list of dict {"x": int, "y": int, "w": int, "h": int, "label": str}
    Mengembalikan objek PIL Image dengan box tergambar.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        label = box.get("label", "lainnya")
        color = BOX_COLORS.get(label, (100, 100, 100))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        # Label background + text
        text = label
        text_bbox = draw.textbbox((0, 0), text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x, max(0, y - text_h - 6), x + text_w + 8, y], fill=color)
        draw.text((x + 4, max(0, y - text_h - 4)), text, fill=(255, 255, 255))

    return img


def compute_iou(box_a, box_b):
    """
    Menghitung Intersection over Union (IoU) antara dua bounding box.
    Setiap box: dict {"x": int, "y": int, "w": int, "h": int}
    Mengembalikan skor 0.0 - 1.0 (1.0 = sempurna sama).
    """
    ax1, ay1, ax2, ay2 = box_a["x"], box_a["y"], box_a["x"] + box_a["w"], box_a["y"] + box_a["h"]
    bx1, by1, bx2, by2 = box_b["x"], box_b["y"], box_b["x"] + box_b["w"], box_b["y"] + box_b["h"]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
