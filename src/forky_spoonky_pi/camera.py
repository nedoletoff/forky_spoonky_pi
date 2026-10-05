"""Воркер камеры: захват, инференс, публикация кадра в общее состояние."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Protocol

import cv2

from . import state as state_module
from .config import AppConfig, CameraConfig
from .detector import Detector
from .logging_setup import get_logger

log = get_logger(__name__)


class FrameSource(Protocol):
    """Общий интерфейс источника кадров (USB или CSI)."""

    def is_opened(self) -> bool: ...

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


class UsbCamera:
    """Камера через V4L2/OpenCV: USB-вебка или CSI в режиме V4L2."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        backend = cv2.CAP_V4L2 if self.config.use_v4l2 else cv2.CAP_ANY
        target = self.config.device or self.config.index
        self._cap = cv2.VideoCapture(target, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, object]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class CsiCamera:
    """Камера Raspberry Pi (CSI) через picamera2/libcamera."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._picam = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - зависит от железа
            raise RuntimeError(
                "Источник 'csi' требует picamera2. Установите на Raspberry Pi: "
                "sudo apt install -y python3-picamera2"
            ) from exc
        self._picam = Picamera2()
        cfg = self._picam.create_preview_configuration(
            main={"size": (self.config.width, self.config.height), "format": "RGB888"}
        )
        self._picam.configure(cfg)
        self._picam.start()

    def is_opened(self) -> bool:
        return self._picam is not None

    def read(self) -> tuple[bool, object]:
        if self._picam is None:
            return False, None
        try:
            rgb = self._picam.capture_array()
        except Exception:  # noqa: BLE001  # pragma: no cover - только на реальном железе
            return False, None
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        if self._picam is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - только на железе
                self._picam.stop()
            self._picam.close()
            self._picam = None


def create_source(config: CameraConfig) -> FrameSource:
    """Фабрика источника кадров по типу из конфига."""
    source = (config.source or "usb").strip().lower()
    if source == "csi":
        return CsiCamera(config)
    if source == "usb":
        return UsbCamera(config)
    raise ValueError(f"Неизвестный источник камеры: {config.source!r} (ожидается 'usb' или 'csi')")


class CameraWorker:
    def __init__(self, config: AppConfig, detector: Detector) -> None:
        self.config = config
        self.detector = detector
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> FrameSource | None:
        """Создаёт и открывает источник; при ошибке логирует и возвращает None."""
        try:
            source = create_source(self.config.camera)
            source.open()
        except Exception as exc:  # noqa: BLE001  # не роняем воркер из-за проблем с камерой
            log.error("Не удалось открыть камеру (%s): %s", self.config.camera.source, exc)
            return None
        return source

    def run(self) -> None:
        source = self._open()
        camera_ok = source is not None and source.is_opened()
        label = self.config.camera.description()
        self._set_status("online" if camera_ok else "no-camera")
        if camera_ok:
            log.info("Камера (%s) открыта", label)
        else:
            log.warning("Камера (%s) недоступна, жду переподключения", label)

        prev_t = time.time()
        smoothed_fps = 0.0

        while not self._stop.is_set():
            if not camera_ok:
                self._publish_placeholder()
                time.sleep(self.config.camera.reconnect_delay)
                self._release(source)
                source = self._open()
                camera_ok = source is not None and source.is_opened()
                self._set_status("online" if camera_ok else "no-camera")
                if camera_ok:
                    log.info("Камера снова доступна")
                continue

            ok, frame = source.read()
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

        self._release(source)
        log.info("Воркер камеры остановлен")

    # ---------- Вспомогательное ----------
    def _release(self, source: FrameSource | None) -> None:
        if source is not None:
            source.release()

    def _set_status(self, status: str) -> None:
        with state_module.lock:
            state_module.state["stats"]["status"] = status

    def _publish_placeholder(self) -> None:
        placeholder = self.detector.make_placeholder("Камера недоступна")
        buf = self.detector.encode_jpeg(placeholder, self.config.camera.jpeg_quality)
        if buf:
            with state_module.lock:
                state_module.state["latest_jpeg"] = buf
                state_module.state["display_bgr"] = placeholder
