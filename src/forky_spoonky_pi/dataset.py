"""Экспорт YOLO-датасета: images/labels, train/val split, data.yaml, zip."""

from __future__ import annotations

import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .config import AppConfig
from .logging_setup import get_logger

log = get_logger(__name__)


class DatasetExporter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dataset_dir: Path = config.resolved_paths.dataset
        self.snap_dir: Path = config.resolved_paths.snapshots

    def _collect_labeled(self) -> list[dict[str, Any]]:
        labeled: list[dict[str, Any]] = []
        for jf in self.snap_dir.glob("*.json"):
            try:
                m = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if m.get("labels"):
                labeled.append(m)
        return labeled

    def _write_split(self, split: str, metas: list[dict]) -> None:
        for m in metas:
            sid = m["id"]
            src = self.snap_dir / f"{sid}.jpg"
            if not src.exists():
                continue
            shutil.copy(src, self.dataset_dir / "images" / split / f"{sid}.jpg")
            lines = [
                f"{int(lab['cls_id'])} "
                f"{float(lab['x']):.6f} {float(lab['y']):.6f} "
                f"{float(lab['w']):.6f} {float(lab['h']):.6f}"
                for lab in m["labels"]
            ]
            (self.dataset_dir / "labels" / split / f"{sid}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    def export(self) -> dict[str, Any]:
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            (self.dataset_dir / sub).mkdir(parents=True, exist_ok=True)

        labeled = self._collect_labeled()
        if not labeled:
            return {"ok": False, "error": "no-labeled"}

        random.seed(42)
        random.shuffle(labeled)
        n_val = max(1, int(len(labeled) * 0.15)) if len(labeled) >= 5 else (1 if len(labeled) > 1 else 0)
        val = labeled[:n_val]
        train = labeled[n_val:]

        self._write_split("train", train)
        self._write_split("val", val)

        yaml_lines = [
            f"path: {self.dataset_dir}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        for i, name in enumerate(self.config.classes.dataset):
            yaml_lines.append(f"  {i}: {name}")
        (self.dataset_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        zip_path = self.dataset_dir / "dataset.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(self.dataset_dir):
                for f in files:
                    if f == "dataset.zip":
                        continue
                    full = Path(root) / f
                    z.write(full, full.relative_to(self.dataset_dir))

        log.info("Датасет экспортирован: train=%d val=%d", len(train), len(val))
        return {"ok": True, "total": len(labeled), "train": len(train), "val": len(val)}
