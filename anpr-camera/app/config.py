from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value


@dataclass(frozen=True)
class Settings:
    camera_host: str = _env("CAMERA_HOST", "192.168.83.22")
    camera_port: int = int(_env("CAMERA_PORT", "80"))
    camera_user: str = _env("CAMERA_USER", "admin")
    camera_password: str = _env("CAMERA_PASSWORD", "")
    camera_rtsp_port: int = int(_env("CAMERA_RTSP_PORT", "554"))
    camera_snapshot_url: str = _env("CAMERA_SNAPSHOT_URL", "")
    camera_rtsp_url: str = _env("CAMERA_RTSP_URL", "")
    recognize_interval: float = float(_env("RECOGNIZE_INTERVAL", "1.2"))
    dedup_seconds: int = int(_env("DEDUP_SECONDS", "20"))
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8080"))
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "data" / "detections.db"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
