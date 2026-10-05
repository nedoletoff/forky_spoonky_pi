"""Работа со снимками: сохранение, список, разметка, переобработка."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from . import state as state_module
from .config import AppConfig
from .detector import Detector, detection_to_label
from .logging_setup import get_logger

log = get_logger(__name__)


class SnapshotStore:
    def __init__(self, config: AppConfig, detector: Detector) -> None:
        self.config = config
        self.detector = detector
        self.store: Path = config.resolved_paths.snapshots

    # ---------- Чтение ----------
    def load_meta(self, sid: str) -> dict[str, Any] | None:
        jf = self.store / f"{sid}.json"
        if not jf.exists():
            return None
        try:
            return json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось прочитать метаданные %s: %s", sid, exc)
            return None

    def image_path(self, sid: str) -> Path | None:
        p = self.store / f"{sid}.jpg"
        return p if p.exists() else None

    def list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for jf in self.store.glob("*.json"):
            try:
                m = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            c = m.get("counts", {})
            items.append(
                {
                    "id": m.get("id", jf.stem),
                    "created": m.get("created", 0),
                    "created_str": m.get("created_str", ""),
                    "fork": c.get("fork", 0),
                    "spoon": c.get("spoon", 0),
                    "unknown": c.get("unknown", 0),
                    "labels_count": len(m.get("labels", [])),
                    "labeled": bool(m.get("labels")),
                }
            )
        items.sort(key=lambda x: x["created"], reverse=True)
        return items

    # ---------- Создание ----------
    def capture(self) -> dict[str, Any] | None:
        """Сохраняет текущий кадр и замораживает поток."""
        with state_module.lock:
            raw_src = state_module.state["raw_processed_bgr"]
            display = state_module.state["display_bgr"]
            if raw_src is None and display is None:
                return None
            raw = (raw_src if raw_src is not None else display).copy()
            drawn = display.copy() if display is not None else raw.copy()
            dets = list(state_module.state["last_detections"])
            counts = {k: state_module.state["stats"].get(k, 0) for k in ("fork", "spoon", "unknown")}
            threshold = state_module.state["stats"]["threshold"]
            status = state_module.state["stats"].get("status", "online")
            infer_ms = state_module.state["stats"].get("infer_ms", 0)

        sid = "snap_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        img_path = self.store / f"{sid}.jpg"
        if not cv2.imwrite(str(img_path), raw, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            log.error("Не удалось записать снимок %s", img_path)
            return None

        height, width = raw.shape[:2]
        labels = [
            label
            for label in (detection_to_label(self.config.classes.dataset, d, width, height) for d in dets)
            if label is not None
        ]
        meta = {
            "id": sid,
            "created": time.time(),
            "created_str": datetime.now().strftime("%d.%m %H:%M:%S"),
            "threshold": threshold,
            "imgsz": self.config.model.imgsz,
            "width": width,
            "height": height,
            "counts": counts,
            "detections": dets,
            "labels": labels,
            "labeled": len(labels) > 0,
        }
        (self.store / f"{sid}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        buf = self.detector.encode_jpeg(drawn, self.config.camera.jpeg_quality)
        with state_module.lock:
            state_module.state["frozen_bgr"] = raw
            state_module.state["display_bgr"] = drawn
            state_module.state["last_detections"] = dets
            if buf:
                state_module.state["latest_jpeg"] = buf
            state_module.state["mode"] = "frozen"
            state_module.state["stats"] = {
                **counts,
                "fps": 0.0,
                "infer_ms": round(infer_ms, 0),
                "status": status,
                "mode": "frozen",
                "threshold": threshold,
            }
        log.info("Снимок сохранён: %s (вилка=%s, ложка=%s)", sid, counts["fork"], counts["spoon"])
        return {"id": sid, "counts": counts}

    # ---------- Обновление ----------
    def save_labels(self, sid: str, raw_labels: list[dict]) -> int | None:
        meta = self.load_meta(sid)
        if meta is None:
            return None
        clean = []
        allowed = set(range(len(self.config.classes.dataset)))
        for label in raw_labels:
            try:
                cls_id = int(label["cls_id"])
                x, y, w, h = float(label["x"]), float(label["y"]), float(label["w"]), float(label["h"])
            except (KeyError, ValueError, TypeError):
                continue
            if cls_id not in allowed:
                continue
            if not (0 <= w <= 1 and 0 <= h <= 1):
                continue
            clean.append(
                {
                    "cls_id": cls_id,
                    "x": max(0.0, min(1.0, x)),
                    "y": max(0.0, min(1.0, y)),
                    "w": w,
                    "h": h,
                }
            )
        meta["labels"] = clean
        meta["labeled"] = len(clean) > 0
        meta["labels_edited_at"] = time.time()
        (self.store / f"{sid}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Метки снимка %s обновлены: %d боксов", sid, len(clean))
        return len(clean)

    def delete(self, sid: str) -> bool:
        removed = False
        for ext in ("jpg", "json"):
            p = self.store / f"{sid}.{ext}"
            if p.exists():
                p.unlink()
                removed = True
        if removed:
            log.info("Снимок удалён: %s", sid)
        return removed

    # ---------- Инференс по снимку ----------
    def detect(self, sid: str, threshold: float) -> dict[str, Any] | None:
        p = self.image_path(sid)
        if p is None:
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        threshold = max(0.05, min(0.95, threshold))
        _, _, infer_ms, dets = self.detector.process_frame(img, threshold)
        height, width = img.shape[:2]
        labels = [
            label
            for label in (detection_to_label(self.config.classes.dataset, d, width, height) for d in dets)
            if label is not None
        ]
        return {"labels": labels, "infer_ms": round(infer_ms, 0), "all_detections": dets}

    def load_to_view(self, sid: str, threshold: float) -> dict[str, Any] | None:
        p = self.image_path(sid)
        if p is None:
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        threshold = max(0.05, min(0.95, threshold))
        drawn, counts, infer_ms, dets = self.detector.process_frame(img, threshold)
        self.detector.overlay_meta(drawn, infer_ms=infer_ms, frozen=True)
        buf = self.detector.encode_jpeg(drawn, self.config.camera.jpeg_quality)
        with state_module.lock:
            status = state_module.state["stats"].get("status", "online")
            state_module.state["frozen_bgr"] = img
            state_module.state["raw_processed_bgr"] = img
            state_module.state["display_bgr"] = drawn
            state_module.state["last_detections"] = dets
            if buf:
                state_module.state["latest_jpeg"] = buf
            state_module.state["mode"] = "frozen"
            state_module.state["stats"] = {
                **counts,
                "fps": 0.0,
                "infer_ms": round(infer_ms, 0),
                "status": status,
                "mode": "frozen",
                "threshold": threshold,
            }
        return {"counts": counts, "infer_ms": round(infer_ms, 0)}

    def reprocess_all(self, threshold: float) -> int:
        """Переобработка всех снимков. Не перетирает ручную разметку."""
        threshold = max(0.05, min(0.95, threshold))
        processed = 0
        for jf in self.store.glob("*.json"):
            sid = jf.stem
            p = self.image_path(sid)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            try:
                m = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            _, counts, _, dets = self.detector.process_frame(img, threshold)
            height, width = img.shape[:2]
            labels = [
                label
                for label in (detection_to_label(self.config.classes.dataset, d, width, height) for d in dets)
                if label is not None
            ]
            if not m.get("labels_edited_at"):
                m["labels"] = labels
                m["labeled"] = len(labels) > 0
            m["counts"] = counts
            m["detections"] = dets
            m["reprocessed_at"] = time.time()
            jf.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
        log.info("Переобработано снимков: %d (порог %.2f)", processed, threshold)
        return processed
