"""Работа с моделью YOLO: загрузка, подмена, экспорт."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .logging_setup import get_logger

if TYPE_CHECKING:
    from ultralytics import YOLO

log = get_logger(__name__)


def _yolo_cls():
    """Ленивая загрузка ultralytics (тяжёлая зависимость, только для инференса)."""
    from ultralytics import YOLO

    return YOLO


class ModelHolder:
    """Держит текущую модель и её метаданные; потокобезопасная подмена."""

    def __init__(self, weights: str = "yolo26n.pt") -> None:
        self._weights = weights
        self.model = self._load(weights)
        self.names = self.model.names

    @staticmethod
    def _load(weights: str) -> YOLO:
        yolo = _yolo_cls()
        path = Path(weights)
        if path.exists():
            log.info("Загружаю модель: %s", path.resolve())
            return yolo(str(path))
        log.info("Загружаю модель по имени (будет скачана при необходимости): %s", weights)
        return yolo(weights)

    @property
    def weights(self) -> str:
        return self._weights

    def promote(self, path: str) -> bool:
        """Заменяет рабочую модель. Возвращает True при успехе."""
        if not Path(path).exists():
            log.error("Не найден файл модели: %s", path)
            return False
        try:
            new_model = _yolo_cls()(str(path))
        except Exception as exc:  # noqa: BLE001
            log.exception("Не удалось загрузить модель %s: %s", path, exc)
            return False
        self.model = new_model
        self.names = new_model.names
        self._weights = path
        log.info("Рабочая модель переключена на %s", path)
        return True

    @property
    def is_custom(self) -> bool:
        """True, если загружена не стандартная nano-модель по имени."""
        return Path(self._weights).exists()

    @property
    def display_name(self) -> str:
        if self.is_custom:
            return Path(self._weights).parent.name or Path(self._weights).stem
        return Path(self._weights).stem
