from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)

DEMO_PLATES = (
    "А123ВС777",
    "К456ЕМ199",
    "М007ОО77",
    "О001ОО77",
    "Т777УТ197",
    "Х999ХХ99",
    "Е001КХ50",
    "В555ОР777",
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_russian_plate(plate: str, width: int = 520, height: int = 112) -> np.ndarray:
    """Рисует макет российского номерного знака для демо-сцены и тестов."""
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=10, outline=(20, 20, 20), width=4)
    region_x = int(width * 0.74)
    draw.rectangle((region_x, 6, width - 8, height - 7), fill=(236, 240, 248))
    draw.line((region_x, 6, region_x, height - 7), fill=(30, 30, 30), width=3)

    compact = plate.replace(" ", "")
    letter = compact[0]
    digits = compact[1:4]
    tail = compact[4:6]
    region = compact[6:]

    font_big = _font(int(height * 0.72))
    font_mid = _font(int(height * 0.52))
    font_sm = _font(int(height * 0.22))

    draw.text((18, height * 0.18), letter, font=font_mid, fill=(15, 15, 15))
    draw.text((int(width * 0.14), height * 0.05), digits, font=font_big, fill=(15, 15, 15))
    draw.text((int(width * 0.48), height * 0.18), tail, font=font_mid, fill=(15, 15, 15))
    draw.text((region_x + 14, height * 0.12), region, font=font_mid, fill=(15, 15, 15))
    draw.text((region_x + 22, height * 0.68), "RUS", font=font_sm, fill=(30, 60, 140))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_demo_frame(t: float, plates: Iterable[str] = DEMO_PLATES) -> np.ndarray:
    """Имитация кадра с камеры наблюдения: дорога и проезжающий номер."""
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # небо / фон
    for y in range(h):
        mix = y / h
        frame[y, :] = (
            int(18 + 20 * mix),
            int(24 + 28 * mix),
            int(32 + 36 * mix),
        )
    cv2.rectangle(frame, (0, 430), (w, h), (42, 46, 52), -1)
    cv2.rectangle(frame, (0, 520), (w, 620), (58, 62, 68), -1)
    for x in range(-80, w, 140):
        shift = int((t * 220) % 140)
        cv2.rectangle(frame, (x + shift, 562), (x + shift + 70, 576), (230, 230, 210), -1)

    plates_list = list(plates)
    idx = int(t // 4) % len(plates_list)
    plate = plates_list[idx]
    local = t % 4
    x = int(-200 + (w + 400) * (local / 4))
    y = 430
    car_w, car_h = 420, 150
    cv2.rectangle(frame, (x, y), (x + car_w, y + car_h), (36, 48, 92), -1)
    cv2.rectangle(frame, (x + 40, y + 20), (x + 190, y + 80), (90, 160, 200), -1)
    cv2.rectangle(frame, (x + 210, y + 20), (x + 340, y + 80), (90, 160, 200), -1)
    cv2.circle(frame, (x + 70, y + car_h), 28, (20, 20, 20), -1)
    cv2.circle(frame, (x + car_w - 70, y + car_h), 28, (20, 20, 20), -1)

    plate_img = draw_russian_plate(plate, width=260, height=64)
    ph, pw = plate_img.shape[:2]
    px, py = x + 80, y + 95
    if 0 <= px < w and 0 <= py < h:
        x2 = min(w, px + pw)
        y2 = min(h, py + ph)
        crop = plate_img[: y2 - py, : x2 - px]
        if crop.size:
            frame[py:y2, px:x2] = crop

    overlay = f"DEMO  {plate}  камера недоступна из этой сети"
    cv2.putText(
        frame,
        overlay,
        (36, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (140, 255, 210),
        2,
        cv2.LINE_AA,
    )
    return frame
