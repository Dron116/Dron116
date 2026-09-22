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
    img = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=10, outline=(18, 18, 18), width=5)
    region_x = int(width * 0.72)
    draw.rectangle((region_x, 8, width - 10, height - 9), fill=(232, 238, 248))
    draw.line((region_x, 8, region_x, height - 9), fill=(20, 20, 20), width=4)

    compact = plate.replace(" ", "")
    letter = compact[0]
    digits = compact[1:4]
    tail = compact[4:6]
    region = compact[6:]

    font_main = _font(int(height * 0.62))
    font_reg = _font(int(height * 0.55))
    font_sm = _font(int(height * 0.18))

    draw.text((16, height * 0.16), letter, font=font_main, fill=(12, 12, 12))
    draw.text((int(width * 0.13), height * 0.12), digits, font=font_main, fill=(12, 12, 12))
    draw.text((int(width * 0.46), height * 0.16), tail, font=font_main, fill=(12, 12, 12))
    draw.text((region_x + 10, height * 0.08), region, font=font_reg, fill=(12, 12, 12))
    draw.text((region_x + 18, height * 0.68), "RUS", font=font_sm, fill=(30, 60, 140))
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
    idx = int(t // 6) % len(plates_list)
    plate = plates_list[idx]
    progress = (t % 6) / 6.0
    car_w, car_h = 520, 190
    x = int(30 + (w - car_w - 60) * progress)
    y = 410
    cv2.rectangle(frame, (x, y), (x + car_w, y + car_h), (36, 48, 92), -1)
    cv2.rectangle(frame, (x + 40, y + 18), (x + 200, y + 78), (90, 160, 200), -1)
    cv2.rectangle(frame, (x + 230, y + 18), (x + 390, y + 78), (90, 160, 200), -1)
    cv2.circle(frame, (x + 80, y + car_h), 30, (20, 20, 20), -1)
    cv2.circle(frame, (x + car_w - 80, y + car_h), 30, (20, 20, 20), -1)

    plate_img = draw_russian_plate(plate, width=420, height=100)
    ph, pw = plate_img.shape[:2]
    px = x + (car_w - pw) // 2
    py = y + 78
    frame[py : py + ph, px : px + pw] = plate_img

    overlay = "DEMO  камера недоступна в этой сети"
    frame = _put_text(frame, overlay, (32, 24), 28, (140, 255, 210))
    return frame


def _put_text(frame: np.ndarray, text: str, xy: tuple[int, int], size: int, color: tuple[int, int, int]) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(xy, text, font=_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
