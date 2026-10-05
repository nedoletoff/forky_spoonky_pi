"""Общее разделяемое состояние приложения (под одним мьютексом)."""

from __future__ import annotations

import threading
from typing import Any

# Замок вокруг изменяемого состояния кадра/статистики.
lock = threading.Lock()

# Замок вокруг инференса и подмены модели.
model_lock = threading.Lock()


def initial_state(threshold: float) -> dict[str, Any]:
    return {
        "mode": "live",
        "latest_jpeg": None,
        "display_bgr": None,
        "raw_processed_bgr": None,
        "frozen_bgr": None,
        "last_detections": [],
        "stats": {
            "fork": 0,
            "spoon": 0,
            "unknown": 0,
            "fps": 0.0,
            "infer_ms": 0.0,
            "status": "init",
            "mode": "live",
            "threshold": threshold,
        },
    }


# Разделяемое состояние; наполняется при старте приложения.
state: dict[str, Any] = initial_state(0.4)

# Состояние процесса обучения.
train_state: dict[str, Any] = {
    "proc": None,
    "log_path": None,
    "name": None,
    "started": None,
    "best": None,
}
