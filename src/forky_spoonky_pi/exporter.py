"""Экспорт фотографий: ZIP всех снимков (с разметкой или без) и отдельные кадры."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2

from .config import AppConfig
from .detector import Detector
from .logging_setup import get_logger

log = get_logger(__name__)


class PhotoExporter:
    def __init__(self, config: AppConfig, detector: Detector) -> None:
        self.config = config
        self.detector = detector
        self.snap_dir: Path = config.resolved_paths.snapshots

    def single_frame(self, sid: str, annotated: bool) -> bytes | None:
        p = self.snap_dir / f"{sid}.jpg"
        if not p.exists():
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        if annotated:
            meta = self._load_meta(sid)
            if meta and meta.get("labels"):
                img = self.detector.draw_yolo_labels(img, meta["labels"])
        return self.detector.encode_jpeg(img, quality=95)

    def zip_all(self, annotated: bool) -> bytes | None:
        """Собирает ZIP со всеми снимками. Возвращает байты архива."""
        sids = sorted({p.stem for p in self.snap_dir.glob("*.jpg")})
        if not sids:
            return None

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for sid in sids:
                data = self.single_frame(sid, annotated)
                if data is None:
                    continue
                suffix = "_annotated" if annotated else ""
                z.writestr(f"{sid}{suffix}.jpg", data)
        buffer.seek(0)
        log.info("Экспортирован ZIP фотографий (%d шт, annotated=%s)", len(sids), annotated)
        return buffer.getvalue()

    def _load_meta(self, sid: str) -> dict | None:
        jf = self.snap_dir / f"{sid}.json"
        if not jf.exists():
            return None
        try:
            return json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
