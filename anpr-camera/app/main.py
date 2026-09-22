from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.camera import CameraClient, DemoSource, encode_jpeg
from app.config import settings
from app.demo import render_demo_frame
from app.detector import Detection, annotate, recognize
from app.history import HistoryStore

log = logging.getLogger("anpr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class Hub:
    def __init__(self) -> None:
        self.camera = CameraClient(settings)
        self.demo = DemoSource(render_demo_frame)
        self.history = HistoryStore()
        self.use_demo = True
        self.latest_detections: list[Detection] = []
        self.annotated_jpeg: bytes | None = None
        self._ws: set[WebSocket] = set()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._stop.clear()
        self.camera.start()
        self.demo.start()
        self._threads = [
            threading.Thread(target=self._watch_source, name="source-watch", daemon=True),
            threading.Thread(target=self._recognize_loop, name="recognize", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.camera.stop()
        self.demo.stop()

    def source_status(self) -> dict[str, Any]:
        live = self.camera.status
        demo = self.demo.status
        active = demo if self.use_demo else live
        return {
            "mode": "demo" if self.use_demo else "live",
            "connected": bool(active.connected and active.last_frame_at),
            "source": active.source,
            "last_error": live.last_error,
            "camera_host": settings.camera_host,
            "width": active.width,
            "height": active.height,
            "last_frame_at": active.last_frame_at,
            "password_configured": bool(settings.camera_password),
        }

    def current_jpeg(self) -> bytes | None:
        if self.annotated_jpeg:
            return self.annotated_jpeg
        src = self.demo if self.use_demo else self.camera
        return src.latest_jpeg()

    def current_frame(self) -> np.ndarray | None:
        src = self.demo if self.use_demo else self.camera
        return src.latest_frame()

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        message = json.dumps(payload, ensure_ascii=False)
        for ws in list(self._ws):
            try:
                await ws.send_text(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._ws.discard(ws)

    def _watch_source(self) -> None:
        while not self._stop.is_set():
            live = self.camera.status
            fresh = live.connected and (time.time() - live.last_frame_at) < 5
            self.use_demo = not fresh
            self._stop.wait(1.0)

    def _recognize_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.current_frame()
            if frame is None:
                self._stop.wait(0.3)
                continue
            try:
                detections = recognize(frame)
                vis = annotate(frame, detections) if detections else frame
                self.annotated_jpeg = encode_jpeg(vis)
                self.latest_detections = detections
                events = []
                for det in detections:
                    saved = self.history.add(
                        plate=det.plate,
                        display=det.display,
                        confidence=det.confidence,
                        raw=det.raw,
                        dedup_seconds=settings.dedup_seconds,
                    )
                    if saved:
                        events.append(saved)
                if events and self._loop:
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast({"type": "detections", "items": events}),
                        self._loop,
                    )
            except Exception:
                log.exception("recognition failed")
            self._stop.wait(settings.recognize_interval)


hub = Hub()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    hub.start(asyncio.get_running_loop())
    yield
    hub.stop()


app = FastAPI(title="ANPR Camera", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return {
        "camera": hub.source_status(),
        "stats": hub.history.stats(),
        "latest": [
            {
                "plate": d.plate,
                "display": d.display,
                "confidence": d.confidence,
                "bbox": d.bbox,
            }
            for d in hub.latest_detections
        ],
    }


@app.get("/api/detections")
async def api_detections(limit: int = 50, q: str = "") -> dict[str, Any]:
    return {"items": hub.history.list(limit=min(limit, 200), query=q)}


@app.delete("/api/detections")
async def api_clear() -> dict[str, str]:
    hub.history.clear()
    return {"ok": "cleared"}


@app.get("/api/stream")
async def api_stream() -> StreamingResponse:
    def gen():
        idle = 0
        while True:
            jpeg = hub.current_jpeg()
            if jpeg:
                idle = 0
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            else:
                idle += 1
                placeholder = _placeholder()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
            time.sleep(0.12 if idle < 50 else 0.4)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/recognize")
async def api_recognize(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "не удалось прочитать изображение"}, status_code=400)
    detections = recognize(frame)
    items = []
    for det in detections:
        saved = hub.history.add(
            plate=det.plate,
            display=det.display,
            confidence=det.confidence,
            raw=det.raw,
            dedup_seconds=0,
        )
        items.append(
            {
                "plate": det.plate,
                "display": det.display,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "saved": saved is not None,
            }
        )
    vis = annotate(frame, detections)
    return JSONResponse({"items": items, "preview": encode_jpeg(vis).hex()[:0], "count": len(items)})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hub._ws.add(ws)
    try:
        await ws.send_json({"type": "hello", "camera": hub.source_status()})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub._ws.discard(ws)


def _placeholder() -> bytes:
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Waiting for camera...",
        (180, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return encode_jpeg(frame)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
