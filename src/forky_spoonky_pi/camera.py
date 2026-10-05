"""Воркер камеры: захват, инференс, публикация кадра в общее состояние."""

from __future__ import annotations

import threading
import time

import cv2

from . import state as state_module
from .config import AppConfig
from .detector import Detector
from .logging_setup import get_logger

log = get_logger(__name__)


class CameraWorker:
    def __init__(self, config: AppConfig, detector: Detector) -> None:
        self.config = config
        self.detector = detector
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> cv2.VideoCapture:
        backend = cv2.CAP_V4L2 if self.config.camera.use_v4l2 else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.config.camera.index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        cap = self._open()
        camera_ok = cap.isOpened()
        self._set_status("online" if camera_ok else "no-camera")
        if camera_ok:
            log.info("Камера %s открыта", self.config.camera.index)
        else:
            log.warning("Камера %s недоступна, жду переподключения", self.config.camera.index)

        prev_t = time.time()
        smoothed_fps = 0.0

        while not self._stop.is_set():
            if not camera_ok:
                self._publish_placeholder(cap)
                time.sleep(self.config.camera.reconnect_delay)
                cap.release()
                cap = self._open()
                camera_ok = cap.isOpened()
                self._set_status("online" if camera_ok else "no-camera")
                if camera_ok:
                    log.info("Камера снова доступна")
                continue

            ok, frame = cap.read()
            if not ok:
                log.warning("Не удалось прочитать кадр — переподключение")
                camera_ok = False
                continue

            with state_module.lock:
                current_mode = state_module.state["mode"]
                threshold = state_module.state["stats"]["threshold"]

            if current_mode == "frozen":
                time.sleep(0.05)
                continue

            drawn, counts, infer_ms, dets = self.detector.process_frame(frame, threshold)

            now = time.time()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            inst_fps = 1.0 / dt
            smoothed_fps = inst_fps if smoothed_fps == 0 else 0.85 * smoothed_fps + 0.15 * inst_fps

            self.detector.overlay_meta(drawn, fps=smoothed_fps, infer_ms=infer_ms)
            buf = self.detector.encode_jpeg(drawn, self.config.camera.jpeg_quality)
            if not buf:
                continue

            with state_module.lock:
                state_module.state["display_bgr"] = drawn
                state_module.state["raw_processed_bgr"] = frame
                state_module.state["last_detections"] = dets
                state_module.state["latest_jpeg"] = buf
                state_module.state["stats"] = {
                    **counts,
                    "fps": round(smoothed_fps, 1),
                    "infer_ms": round(infer_ms, 0),
                    "status": "online",
                    "mode": "live",
                    "threshold": threshold,
                }

            sleep_for = 1.0 / self.config.camera.fps - (time.time() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        log.info("Воркер камеры остановлен")

    # ---------- Вспомогательное ----------
    def _set_status(self, status: str) -> None:
        with state_module.lock:
            state_module.state["stats"]["status"] = status

    def _publish_placeholder(self, cap: cv2.VideoCapture) -> None:
        placeholder = self.detector.make_placeholder("Камера недоступна")
        buf = self.detector.encode_jpeg(placeholder, self.config.camera.jpeg_quality)
        if buf:
            with state_module.lock:
                state_module.state["latest_jpeg"] = buf
                state_module.state["display_bgr"] = placeholder
