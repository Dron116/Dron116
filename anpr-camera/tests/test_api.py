from fastapi.testclient import TestClient

from app.demo import draw_russian_plate
from app.main import app


def test_index_and_status():
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "ANPR" in home.text
        status = client.get("/api/status")
        assert status.status_code == 200
        payload = status.json()
        assert payload["camera"]["camera_host"] == "192.168.83.22"
        assert "stats" in payload


def test_detections_endpoint_and_upload():
    with TestClient(app) as client:
        listed = client.get("/api/detections")
        assert listed.status_code == 200
        assert "items" in listed.json()

        import cv2

        image = draw_russian_plate("К456ЕМ199", width=640, height=140)
        ok, buf = cv2.imencode(".png", image)
        assert ok
        response = client.post(
            "/api/recognize",
            files={"file": ("plate.png", buf.tobytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "items" in body
