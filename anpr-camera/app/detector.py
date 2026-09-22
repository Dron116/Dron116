from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from app.plates import candidates_from_ocr, format_plate

log = logging.getLogger("anpr.detector")

TESSERACT_CFG = (
    "--oem 3 --psm 7 "
    "-c tessedit_char_whitelist=ABEKMHOPCTYXАВЕКМНОРСТУХ0123456789"
)


@dataclass
class Detection:
    plate: str
    display: str
    confidence: float
    bbox: tuple[int, int, int, int]
    raw: str


def recognize(frame: np.ndarray) -> list[Detection]:
    if frame is None or frame.size == 0:
        return []
    regions = _plate_regions(frame)
    detections: list[Detection] = []
    seen: set[str] = set()

    for x, y, w, h in regions:
        roi = frame[y : y + h, x : x + w]
        raw, conf = _ocr_plate(roi)
        for plate in candidates_from_ocr(raw):
            if plate in seen:
                continue
            seen.add(plate)
            detections.append(
                Detection(
                    plate=plate,
                    display=format_plate(plate),
                    confidence=conf,
                    bbox=(x, y, w, h),
                    raw=raw,
                )
            )

    if not detections:
        raw, conf = _ocr_plate(frame)
        for plate in candidates_from_ocr(raw):
            h, w = frame.shape[:2]
            detections.append(
                Detection(
                    plate=plate,
                    display=format_plate(plate),
                    confidence=max(conf * 0.7, 0.35),
                    bbox=(0, 0, w, h),
                    raw=raw,
                )
            )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (80, 230, 160), 3)
        label = f"{det.display}  {int(det.confidence * 100)}%"
        ty = max(28, y - 10)
        cv2.putText(
            out,
            label,
            (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (80, 230, 160),
            2,
            cv2.LINE_AA,
        )
    return out


def _plate_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad = np.abs(grad)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    min_area = (w * h) * 0.002
    max_area = (w * h) * 0.35
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < 12 or bw < 40:
            continue
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        ratio = bw / float(bh)
        if ratio < 1.8 or ratio > 7.5:
            continue
        pad_x = int(bw * 0.06)
        pad_y = int(bh * 0.18)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:6]


def _ocr_plate(roi: np.ndarray) -> tuple[str, float]:
    prepared = _prepare_roi(roi)
    try:
        data = pytesseract.image_to_data(
            prepared,
            lang="eng+rus",
            config=TESSERACT_CFG,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as exc:
        log.warning("tesseract failed: %s", exc)
        return "", 0.0

    texts = []
    confs = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        token = (text or "").strip()
        if not token:
            continue
        try:
            score = float(conf)
        except (TypeError, ValueError):
            score = -1
        if score < 0:
            continue
        texts.append(token)
        confs.append(score / 100.0)
    raw = "".join(texts) if texts else pytesseract.image_to_string(
        prepared, lang="eng+rus", config=TESSERACT_CFG
    )
    confidence = float(sum(confs) / len(confs)) if confs else 0.45
    return raw, confidence


def _prepare_roi(roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    h, w = gray.shape[:2]
    scale = 72 / max(h, 1)
    if scale > 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # номера обычно тёмные на светлом; если инверсия даёт больше «белого», оставляем как есть
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary
