"""Фабрика Flask-приложения и запуск воркеров."""

from __future__ import annotations

import threading

from flask import Flask

from ..camera import CameraWorker
from ..config import AppConfig
from ..detector import Detector
from ..logging_setup import get_logger
from ..model import ModelHolder

log = get_logger(__name__)


def create_app(config: AppConfig) -> Flask:
    from .routes import bp

    app = Flask(__name__)
    app.config["FORKY"] = config

    holder = ModelHolder(config.model.weights)
    detector = Detector(config, holder)
    camera = CameraWorker(config, detector)

    app.extensions["forky_holder"] = holder
    app.extensions["forky_detector"] = detector
    app.extensions["forky_camera"] = camera

    threading.Thread(target=camera.run, name="camera-worker", daemon=True).start()
    log.info("Веб-сервер инициализирован")

    app.register_blueprint(bp)
    return app
