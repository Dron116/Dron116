from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote

import cv2
import httpx
import numpy as np

from app.config import Settings, settings

log = logging.getLogger("anpr.camera")

SNAPSHOT_PATHS = (
    "/ISAPI/Streaming/channels/101/picture",
    "/ISAPI/Streaming/channels/1/picture",
    "/cgi-bin/snapshot.cgi",
    "/cgi-bin/snapshot.cgi?channel=1",
    "/cgi-bin/snapshot.cgi?chn=1&u={user}&p={password}",
    "/webcapture.jpg?command=snap&channel=1",
    "/webcapture.jpg?command=snap&channel=1&user={user}&password={password}",
    "/cgi-bin/hi3510/snap.cgi?&-getpic",
    "/snapshot.cgi",
    "/snapshot.jpg",
    "/snap.jpg",
    "/tmpfs/auto.jpg",
    "/image/jpeg.cgi",
    "/onvifsnapshot/media_profile1.jpg",
    "/jpg/image.jpg",
    "/cgi-bin/currentpic.cgi",
)

RTSP_PATHS = (
    "/Streaming/Channels/101",
    "/Streaming/Channels/1",
    "/cam/realmonitor?channel=1&subtype=0",
    "/h264/ch1/main/av_stream",
    "/h264Preview_01_main",
    "/stream1",
    "/live/ch00_0",
    "/unicast/c1/s0/live",
    "/user={user}_password={password}_channel=1_stream=0.sdp",
    "/1/1",
)


@dataclass
class CameraStatus:
    mode: str = "idle"
    connected: bool = False
    source: str = ""
    last_error: str = ""
    last_frame_at: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class FrameStore:
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: np.ndarray | None = None
    jpeg: bytes | None = None
    captured_at: float = 0.0


class CameraClient:
    """Забирает кадры с IP-камеры (HTTP snapshot или RTSP) и хранит последний JPEG."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self.status = CameraStatus()
        self.store = FrameStore()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: str | None = None
        self._kind: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def latest_frame(self) -> np.ndarray | None:
        with self.store.lock:
            if self.store.frame is None:
                return None
            return self.store.frame.copy()

    def latest_jpeg(self) -> bytes | None:
        with self.store.lock:
            return self.store.jpeg

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._source:
                    self.status.mode = "connecting"
                    self._discover()
                if self._kind == "http" and self._source:
                    ok = self._pull_http(self._source)
                    if ok:
                        self._stop.wait(0.15)
                elif self._kind == "rtsp" and self._source:
                    ok = self._pull_rtsp(self._source)
                else:
                    ok = False
                if not ok:
                    self.status.connected = False
                    self._source = None
                    self._kind = None
                    self._stop.wait(2.0)
            except Exception as exc:  # noqa: BLE001 — цикл не должен падать
                log.exception("camera loop failed")
                self.status.last_error = str(exc)
                self.status.connected = False
                self._source = None
                self._kind = None
                self._stop.wait(2.0)

    def _discover(self) -> None:
        cfg = self.cfg
        if cfg.camera_snapshot_url:
            if self._pull_http(cfg.camera_snapshot_url):
                self._source = cfg.camera_snapshot_url
                self._kind = "http"
                return
        if cfg.camera_rtsp_url:
            if self._open_rtsp(cfg.camera_rtsp_url):
                self._source = cfg.camera_rtsp_url
                self._kind = "rtsp"
                return

        host = cfg.camera_host
        http_open = _port_open(host, cfg.camera_port)
        rtsp_open = _port_open(host, cfg.camera_rtsp_port)
        if not http_open and not rtsp_open:
            self.status.last_error = (
                f"Камера {host} не отвечает в этой сети. "
                "Запустите приложение рядом с камерой и укажите CAMERA_PASSWORD."
            )
            self.status.connected = False
            self._stop.wait(1.5)
            return

        user = quote(cfg.camera_user, safe="")
        password = quote(cfg.camera_password, safe="")
        auth = (cfg.camera_user, cfg.camera_password)

        if http_open:
            for path in SNAPSHOT_PATHS:
                url = f"http://{host}:{cfg.camera_port}{path.format(user=user, password=password)}"
                if self._pull_http(url, auth=auth):
                    self._source = url
                    self._kind = "http"
                    self.status.source = url.split("?")[0]
                    return

        if rtsp_open:
            rtsp_auth = f"{user}:{password}@" if cfg.camera_user else ""
            for path in RTSP_PATHS:
                url = (
                    f"rtsp://{rtsp_auth}{host}:{cfg.camera_rtsp_port}"
                    f"{path.format(user=user, password=password)}"
                )
                if self._open_rtsp(url):
                    self._source = url
                    self._kind = "rtsp"
                    self.status.source = f"rtsp://{host}:{cfg.camera_rtsp_port}{path.split('?')[0]}"
                    return

        self.status.last_error = (
            f"Камера {host} недоступна. Проверьте сеть, логин/пароль и URL потока."
        )
        self.status.connected = False

    def _pull_http(self, url: str, auth: tuple[str, str] | None = None) -> bool:
        auth_tuple = auth or (self.cfg.camera_user, self.cfg.camera_password)
        for auth_mode in ("digest", "basic"):
            try:
                auth_obj: httpx.Auth | tuple[str, str]
                if auth_mode == "digest":
                    auth_obj = httpx.DigestAuth(*auth_tuple)
                else:
                    auth_obj = auth_tuple
                with httpx.Client(timeout=2.0, follow_redirects=True) as client:
                    response = client.get(url, auth=auth_obj)
                if response.status_code != 200:
                    continue
                image = _decode_image(response.content)
                if image is None:
                    continue
                self._publish(image)
                self.status.connected = True
                self.status.mode = "live"
                self.status.source = url.split("?")[0]
                self.status.last_error = ""
                return True
            except Exception as exc:  # noqa: BLE001
                self.status.last_error = str(exc)
        return False

    def _open_rtsp(self, url: str) -> bool:
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            return False
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            return False
        self._publish(frame)
        self.status.connected = True
        self.status.mode = "live"
        self.status.last_error = ""
        return True

    def _pull_rtsp(self, url: str) -> bool:
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            return False
        idle = 0
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                idle += 1
                if idle > 30:
                    capture.release()
                    return False
                time.sleep(0.05)
                continue
            idle = 0
            self._publish(frame)
            # не крутить RTSP на полной частоте — достаточно ~8 к/с для UI
            time.sleep(0.08)
        capture.release()
        return True

    def _publish(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        jpeg = encode_jpeg(frame)
        with self.store.lock:
            self.store.frame = frame
            self.store.jpeg = jpeg
            self.store.captured_at = time.time()
        self.status.last_frame_at = time.time()
        self.status.width = w
        self.status.height = h
        self.status.connected = True


def _port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _decode_image(data: bytes) -> np.ndarray | None:
    if not data or len(data) < 32:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("не удалось закодировать JPEG")
    return buf.tobytes()


class DemoSource:
    """Локальный источник кадров, если камера недоступна (для разработки и UI)."""

    def __init__(self, renderer: Callable[[float], np.ndarray]) -> None:
        self.renderer = renderer
        self.store = FrameStore()
        self.status = CameraStatus(mode="demo", connected=True, source="demo")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="demo-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def latest_frame(self) -> np.ndarray | None:
        with self.store.lock:
            return None if self.store.frame is None else self.store.frame.copy()

    def latest_jpeg(self) -> bytes | None:
        with self.store.lock:
            return self.store.jpeg

    def _run(self) -> None:
        t0 = time.time()
        while not self._stop.is_set():
            frame = self.renderer(time.time() - t0)
            jpeg = encode_jpeg(frame, quality=78)
            h, w = frame.shape[:2]
            with self.store.lock:
                self.store.frame = frame
                self.store.jpeg = jpeg
                self.store.captured_at = time.time()
            self.status.last_frame_at = time.time()
            self.status.width = w
            self.status.height = h
            self._stop.wait(0.12)
