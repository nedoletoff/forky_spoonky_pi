"""Инференс и отрисовка: обработка кадра, YOLO-конвертации, метки."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import state as state_module
from .config import AppConfig
from .logging_setup import get_logger

if TYPE_CHECKING:
    from .model import ModelHolder

log = get_logger(__name__)


class Detector:
    def __init__(self, config: AppConfig, holder: ModelHolder) -> None:
        self.config = config
        self.holder = holder
        self.dataset_classes = config.classes.dataset
        self.names_ru = config.classes.names_ru
        self.colors_bgr = {0: (46, 160, 67), 1: (34, 101, 190)}

    # ---------- Отрисовка ----------
    def make_placeholder(self, text: str) -> np.ndarray:
        frame = np.zeros(
            (self.config.camera.height, self.config.camera.width, 3), dtype=np.uint8
        )
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(
            frame,
            text,
            ((self.config.camera.width - tw) // 2, (self.config.camera.height + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (60, 60, 220),
            2,
            cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def draw_label(frame: np.ndarray, x1: int, y1: int, text: str, color: tuple) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_top = max(y1 - th - 12, 0)
        cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y_top + th + 10), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, y_top + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_yolo_labels(self, img: np.ndarray, labels: list[dict]) -> np.ndarray:
        out = img.copy()
        height, width = out.shape[:2]
        for label in labels:
            try:
                cls_id = int(label["cls_id"])
            except (KeyError, ValueError, TypeError):
                continue
            if cls_id not in self.names_ru:
                continue
            x1 = int((label["x"] - label["w"] / 2) * width)
            y1 = int((label["y"] - label["h"] / 2) * height)
            x2 = int((label["x"] + label["w"] / 2) * width)
            y2 = int((label["y"] + label["h"] / 2) * height)
            color = self.colors_bgr.get(cls_id, (150, 150, 150))
            self.draw_label(out, x1, y1, self.names_ru[cls_id], color)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        return out

    def det_to_label(self, det: dict, img_w: int, img_h: int) -> dict | None:
        if det["label"] not in self.dataset_classes:
            return None
        x1, y1, x2, y2 = det["xyxy"]
        return {
            "cls_id": self.dataset_classes.index(det["label"]),
            "x": ((x1 + x2) / 2) / img_w,
            "y": ((y1 + y2) / 2) / img_h,
            "w": (x2 - x1) / img_w,
            "h": (y2 - y1) / img_h,
        }

    # ---------- Инференс ----------
    def process_frame(self, raw_bgr: np.ndarray, threshold: float) -> tuple[np.ndarray, dict, float, list]:
        t0 = time.time()
        with state_module.model_lock:
            results = self.holder.model(
                raw_bgr, imgsz=self.config.model.imgsz, verbose=False, device=self.config.model.device
            )[0]
            names = self.holder.names
        infer_ms = (time.time() - t0) * 1000.0

        drawn = raw_bgr.copy()
        counts = {"fork": 0, "spoon": 0, "unknown": 0}
        detections: list[dict] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                continue
            cls_id = int(box.cls[0])
            label = names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "fork":
                display, color, key = "Вилка", self.colors_bgr[0], "fork"
            elif label == "spoon":
                display, color, key = "Ложка", self.colors_bgr[1], "spoon"
            else:
                display, color, key = "Непонятно", (60, 60, 200), "unknown"

            counts[key] += 1
            detections.append(
                {
                    "label": label,
                    "display": display,
                    "conf": round(conf, 3),
                    "cls_id": cls_id,
                    "xyxy": [x1, y1, x2, y2],
                }
            )
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            self.draw_label(drawn, x1, y1, f"{display} {conf:.2f}", color)

        return drawn, counts, infer_ms, detections

    # ---------- Метрики кадра ----------
    @staticmethod
    def overlay_meta(
        frame: np.ndarray, fps: float | None = None, infer_ms: float | None = None, frozen: bool = False
    ) -> None:
        y = 28
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
            y += 28
        if infer_ms is not None:
            cv2.putText(frame, f"Infer: {infer_ms:.0f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
            y += 28
        if frozen:
            cv2.putText(frame, "SNAPSHOT", (frame.shape[1] - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2, cv2.LINE_AA)

    @staticmethod
    def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> bytes | None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # ---------- Экспорт модели ----------
    def export_model(self, path: str, fmt: str = "ncnn") -> str | None:
        """Экспортирует модель в заданный формат. Возвращает путь к артефакту."""
        from ultralytics import YOLO

        try:
            with state_module.model_lock:
                out = YOLO(str(path)).export(format=fmt, imgsz=self.config.model.imgsz)
            log.info("Модель %s экспортирована (%s): %s", path, fmt, out)
            return str(out)
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка экспорта модели %s (%s): %s", path, fmt, exc)
            return None


def detection_to_label(dataset_classes: list[str], det: dict, img_w: int, img_h: int) -> dict | None:
    """Утилита вне класса (для dataset/training-модулей)."""
    if det["label"] not in dataset_classes:
        return None
    x1, y1, x2, y2 = det["xyxy"]
    return {
        "cls_id": dataset_classes.index(det["label"]),
        "x": ((x1 + x2) / 2) / img_w,
        "y": ((y1 + y2) / 2) / img_h,
        "w": (x2 - x1) / img_w,
        "h": (y2 - y1) / img_h,
    }


__all__ = ["Detector", "detection_to_label"]
