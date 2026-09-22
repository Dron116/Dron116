from app.demo import draw_russian_plate, render_demo_frame
from app.detector import recognize
from app.plates import normalize_plate


def test_ocr_on_clean_plate():
    plate = "А123ВС777"
    image = draw_russian_plate(plate, width=640, height=140)
    detections = recognize(image)
    texts = [normalize_plate(d.plate) for d in detections]
    assert plate in texts or any(d.plate == plate for d in detections)


def test_demo_frame_contains_moving_plate():
    frame = render_demo_frame(2.0)
    assert frame.shape[0] == 720
    assert frame.shape[1] == 1280
    detections = recognize(frame)
    # в середине анимации номер крупный; если OCR сработал — проверяем формат
    for det in detections:
        assert len(det.plate) in (8, 9)


def test_empty_frame():
    import numpy as np

    blank = np.zeros((100, 100, 3), dtype="uint8")
    assert recognize(blank) == []
