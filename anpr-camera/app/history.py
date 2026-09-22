from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from app.config import settings


class HistoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT NOT NULL,
                    display TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    raw TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_plate_time ON detections(plate, created_at DESC)"
            )
            conn.commit()

    def add(
        self,
        plate: str,
        display: str,
        confidence: float,
        raw: str,
        dedup_seconds: int,
    ) -> dict[str, Any] | None:
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, created_at FROM detections
                WHERE plate = ? ORDER BY created_at DESC LIMIT 1
                """,
                (plate,),
            ).fetchone()
            if row and now - float(row["created_at"]) < dedup_seconds:
                return None
            cur = conn.execute(
                """
                INSERT INTO detections (plate, display, confidence, raw, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (plate, display, confidence, raw, now),
            )
            conn.commit()
            return {
                "id": int(cur.lastrowid),
                "plate": plate,
                "display": display,
                "confidence": confidence,
                "raw": raw,
                "created_at": now,
            }

    def list(self, limit: int = 50, query: str = "") -> list[dict[str, Any]]:
        sql = "SELECT id, plate, display, confidence, raw, created_at FROM detections"
        args: list[Any] = []
        if query:
            sql += " WHERE plate LIKE ? OR display LIKE ?"
            like = f"%{query.replace(' ', '').upper()}%"
            args.extend([like, f"%{query}%"])
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()["c"]
            unique = conn.execute("SELECT COUNT(DISTINCT plate) AS c FROM detections").fetchone()["c"]
            last = conn.execute(
                "SELECT plate, display, created_at FROM detections ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return {
            "total": int(total),
            "unique": int(unique),
            "last": dict(last) if last else None,
        }

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM detections")
            conn.commit()
