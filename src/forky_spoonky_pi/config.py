"""Загрузка конфигурации из config.yaml с переопределением через переменные окружения.

Приоритет: значения по умолчанию < config.yaml < переменные окружения FORKY_*.

Переопределения задаются переменными вида FORKY_<СЕКЦИЯ>__<КЛЮЧ>. Например,
FORKY_SERVER__PORT=9000 переопределит server.port. Это позволяет не держать
секреты в config.yaml и удобно для systemd EnvironmentFile=.env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
ENV_PATH = BASE_DIR / ".env"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class CameraConfig:
    # Источник камеры: "usb" (V4L2 через OpenCV) или "csi" (Raspberry Pi, picamera2).
    source: str = "usb"
    index: int = 0
    # Путь к устройству, например /dev/video0. Если пусто — используется index.
    device: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    use_v4l2: bool = True
    jpeg_quality: int = 80
    reconnect_delay: float = 1.0

    def description(self) -> str:
        """Человекочитаемое описание источника для логов."""
        if (self.source or "usb").strip().lower() == "csi":
            return "csi"
        return self.device or str(self.index)


@dataclass
class ModelConfig:
    weights: str = "yolo26n.pt"
    imgsz: int = 416
    confidence: float = 0.4
    device: str = "cpu"


@dataclass
class ClassesConfig:
    dataset: list[str] = field(default_factory=lambda: ["fork", "spoon"])
    names_ru: dict[int, str] = field(default_factory=lambda: {0: "Вилка", 1: "Ложка"})


@dataclass
class PathsConfig:
    data_dir: str = "data"

    def resolve(self) -> ResolvedPaths:
        root = (BASE_DIR / self.data_dir).resolve()
        return ResolvedPaths(
            data=root,
            snapshots=root / "snapshots",
            dataset=root / "dataset",
            runs=root / "runs",
            logs=root / "logs",
            weights=root / "weights",
        )


@dataclass
class LoggingConfig:
    level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5


@dataclass
class TrainingConfig:
    enabled: bool = False
    epochs: int = 30
    imgsz: int = 416
    batch: int = 4
    workers: int = 2
    patience: int = 20
    device: str = "auto"


@dataclass
class ResolvedPaths:
    data: Path
    snapshots: Path
    dataset: Path
    runs: Path
    logs: Path
    weights: Path

    def ensure(self) -> ResolvedPaths:
        for p in (self.data, self.snapshots, self.dataset, self.runs, self.logs, self.weights):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    classes: ClassesConfig = field(default_factory=ClassesConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    base_dir: Path = BASE_DIR

    @property
    def resolved_paths(self) -> ResolvedPaths:
        return self.paths.resolve()


def _coerce(value: str) -> Any:
    """Приводит строку из env к bool/int/float, если это возможно."""
    low = value.strip().lower()
    if low in ("true", "yes", "1", "on"):
        return True
    if low in ("false", "no", "0", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    prefix = "FORKY_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        if not path:
            continue
        node = raw
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[path[-1]] = _coerce(value)
    return raw


def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv(ENV_PATH, override=False)

    path = config_path or CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    raw = _apply_env_overrides(raw)

    server = ServerConfig(**(raw.get("server") or {}))
    camera = CameraConfig(**(raw.get("camera") or {}))
    model = ModelConfig(**(raw.get("model") or {}))
    classes_raw = raw.get("classes") or {}
    classes = ClassesConfig(
        dataset=list(classes_raw.get("dataset") or ["fork", "spoon"]),
        names_ru={int(k): v for k, v in (classes_raw.get("names_ru") or {0: "Вилка", 1: "Ложка"}).items()},
    )
    paths = PathsConfig(**(raw.get("paths") or {}))
    logging_cfg = LoggingConfig(**(raw.get("logging") or {}))
    training = TrainingConfig(**(raw.get("training") or {}))

    return AppConfig(
        server=server,
        camera=camera,
        model=model,
        classes=classes,
        paths=paths,
        logging=logging_cfg,
        training=training,
    )
