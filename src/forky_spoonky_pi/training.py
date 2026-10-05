"""Дообучение модели: сабпроцесс yolo train, парсинг метрик, ETA.

Обучение предполагается на ноутбуке с дискретной GPU. На Raspberry Pi
обучение обычно отключено (training.enabled=false), а готовый best.pt
экспортируется в NCNN скриптом scripts/export_ncnn.py и переносится на Pi.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import state as state_module
from .config import AppConfig
from .logging_setup import get_logger

log = get_logger(__name__)


class Trainer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runs_dir: Path = config.resolved_paths.runs

    # ---------- Запуск ----------
    def start(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self.config.training.enabled:
            return {"ok": False, "error": "training-disabled"}

        proc = state_module.train_state["proc"]
        if proc is not None and proc.poll() is None:
            return {"ok": False, "error": "already-running"}

        data_yaml = self.config.resolved_paths.dataset / "data.yaml"
        if not data_yaml.exists():
            return {"ok": False, "error": "no-dataset"}

        base = params.get("base_model") or self.config.model.weights
        epochs = max(1, int(params.get("epochs", self.config.training.epochs)))
        imgsz = int(params.get("imgsz", self.config.training.imgsz))
        batch = max(1, int(params.get("batch", self.config.training.batch)))
        device = params.get("device", self.config.training.device)

        name = "train_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.runs_dir / f"{name}.log"
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

        if not Path(base).is_absolute() and not Path(base).exists():
            base_expr = f'"{base}"'
        else:
            base_expr = repr(str(Path(base).resolve()))

        code = f"""
from ultralytics import YOLO
YOLO({base_expr}).train(
    data=r"{data_yaml}",
    epochs={epochs},
    imgsz={imgsz},
    batch={batch},
    device={device!r},
    workers={self.config.training.workers},
    patience={self.config.training.patience},
    project=r"{self.runs_dir}",
    name=r"{name}",
    exist_ok=True,
    plots=False,
    val=True,
)
"""
        popen = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(self.config.base_dir),
        )
        state_module.train_state.update(
            {
                "proc": popen,
                "log_path": log_path,
                "name": name,
                "started": time.time(),
                "best": None,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
            }
        )
        log.info(
            "Запущено обучение %s: base=%s epochs=%d imgsz=%d batch=%d device=%s",
            name, base, epochs, imgsz, batch, device,
        )
        return {"ok": True, "name": name, "epochs": epochs}

    def stop(self) -> bool:
        proc = state_module.train_state["proc"]
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.warning("Обучение остановлено пользователем")
        return True

    # ---------- Статус и метрики ----------
    def _results_csv(self) -> Path | None:
        name = state_module.train_state.get("name")
        if not name:
            return None
        p = self.runs_dir / name / "results.csv"
        return p if p.exists() else None

    def _parse_metrics(self) -> list[dict[str, Any]]:
        csv_path = self._results_csv()
        if csv_path is None:
            return []
        rows: list[dict[str, Any]] = []
        try:
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    clean: dict[str, Any] = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        k = key.strip()
                        try:
                            clean[k] = float(str(value).strip())
                        except (ValueError, TypeError):
                            clean[k] = value
                    rows.append(clean)
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось распарсить results.csv: %s", exc)
        return rows

    def metrics(self) -> dict[str, Any]:
        rows = self._parse_metrics()
        epochs = state_module.train_state.get("epochs") or self.config.training.epochs

        series: dict[str, list[float]] = {
            "box_loss": [],
            "cls_loss": [],
            "dfl_loss": [],
            "mAP50": [],
            "mAP50-95": [],
            "precision": [],
            "recall": [],
        }
        matched = {
            "box_loss": "train/box_loss",
            "cls_loss": "train/cls_loss",
            "dfl_loss": "train/dfl_loss",
            "mAP50": "metrics/mAP50(B)",
            "mAP50-95": "metrics/mAP50-95(B)",
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
        }
        for row in rows:
            for out_key, csv_key in matched.items():
                value = row.get(csv_key)
                if isinstance(value, (int, float)):
                    series[out_key].append(round(float(value), 5))

        current_epoch = len(rows)
        elapsed = time.time() - (state_module.train_state.get("started") or time.time())
        avg_epoch = elapsed / current_epoch if current_epoch else 0.0
        remaining = max(epochs - current_epoch, 0)
        eta = avg_epoch * remaining if avg_epoch else None

        return {
            "series": series,
            "current_epoch": current_epoch,
            "epochs": epochs,
            "avg_epoch_seconds": round(avg_epoch, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "progress": round(min(current_epoch / epochs, 1.0), 4) if epochs else 0.0,
        }

    def status(self) -> dict[str, Any]:
        proc = state_module.train_state["proc"]
        running = proc is not None and proc.poll() is None

        log_tail = ""
        lp = state_module.train_state.get("log_path")
        if lp and Path(lp).exists():
            try:
                text = Path(lp).read_text(errors="ignore", encoding="utf-8")
                log_tail = text[-6000:]
            except Exception:  # noqa: BLE001
                pass

        best = None
        name = state_module.train_state.get("name")
        if name:
            cand = self.runs_dir / name / "weights" / "best.pt"
            if cand.exists():
                best = str(cand)
                state_module.train_state["best"] = best

        metrics = self.metrics()
        return {
            "enabled": self.config.training.enabled,
            "running": running,
            "log": log_tail,
            "name": name,
            "best": best,
            "started": state_module.train_state.get("started"),
            "current_epoch": metrics["current_epoch"],
            "epochs": metrics["epochs"],
            "eta_seconds": metrics["eta_seconds"],
            "progress": metrics["progress"],
        }
