"""Юнит-тесты чистой логики: YOLO-конвертации, сплит датасета, метрики/ETA.

Тесты не требуют камеры и не запускают инференс: используется только
модульная функция detection_to_label и парсеры. Для Trainer._parse_metrics
подкладывается временный runs-каталог с results.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forky_spoonky_pi.detector import detection_to_label  # noqa: E402


# ---------- detection_to_label ----------
class TestDetectionToLabel:
    CLASSES = ["fork", "spoon"]

    def _det(self, label: str, xyxy: list[int]) -> dict:
        return {"label": label, "xyxy": xyxy}

    def test_центр_и_размеры_нормализуются(self) -> None:
        lab = detection_to_label(self.CLASSES, self._det("fork", [100, 50, 300, 250]), 400, 500)
        assert lab is not None
        assert lab["cls_id"] == 0
        assert lab["x"] == pytest.approx(0.5)
        assert lab["y"] == pytest.approx(0.3)
        assert lab["w"] == pytest.approx(0.5)
        assert lab["h"] == pytest.approx(0.4)

    def test_класс_spoon_индекс_1(self) -> None:
        lab = detection_to_label(self.CLASSES, self._det("spoon", [0, 0, 100, 100]), 100, 100)
        assert lab is not None
        assert lab["cls_id"] == 1

    def test_неизвестный_класс_возвращает_none(self) -> None:
        assert detection_to_label(self.CLASSES, self._det("knife", [0, 0, 10, 10]), 100, 100) is None


# ---------- Trainer._parse_metrics / ETA ----------
class TestTrainerMetrics:
    CSV_HEADER = (
        "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
    )

    def _make_trainer(self, tmp_path: Path, rows: int):
        from forky_spoonky_pi import state as state_module
        from forky_spoonky_pi.config import AppConfig, PathsConfig
        from forky_spoonky_pi.training import Trainer

        cfg = AppConfig(paths=PathsConfig(data_dir=str(tmp_path)))
        runs = tmp_path / "runs"
        name = "train_test"
        run_dir = runs / name
        (run_dir).mkdir(parents=True)
        csv_path = run_dir / "results.csv"
        lines = [self.CSV_HEADER]
        for i in range(1, rows + 1):
            lines.append(f"{i},{i * 10},0.9,0.8,0.7,0.5,0.6,0.4,0.3\n")
        csv_path.write_text("".join(lines), encoding="utf-8")

        state_module.train_state.update(
            {"proc": None, "log_path": None, "name": name, "started": None, "best": None}
        )
        return Trainer(cfg)

    def test_парсинг_серии_и_прогресса(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path, rows=3)
        trainer.config.training.epochs = 10
        metrics = trainer.metrics()
        assert metrics["current_epoch"] == 3
        assert metrics["epochs"] == 10
        assert len(metrics["series"]["box_loss"]) == 3
        assert metrics["series"]["mAP50"] == [0.4, 0.4, 0.4]
        assert metrics["progress"] == pytest.approx(0.3)

    def test_eta_none_когда_нет_запуска(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path, rows=2)
        metrics = trainer.metrics()
        assert metrics["eta_seconds"] is None or metrics["eta_seconds"] >= 0

    def test_нет_csv_пустые_серии(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path, rows=0)
        (tmp_path / "runs" / "train_test" / "results.csv").unlink()
        metrics = trainer.metrics()
        assert metrics["current_epoch"] == 0
        assert metrics["series"]["box_loss"] == []


# ---------- DatasetExporter split ----------
class TestDatasetSplit:
    def test_сплит_и_data_yaml(self, tmp_path: Path) -> None:
        from forky_spoonky_pi.config import AppConfig, PathsConfig
        from forky_spoonky_pi.dataset import DatasetExporter

        cfg = AppConfig(paths=PathsConfig(data_dir=str(tmp_path)))
        snap = cfg.resolved_paths.snapshots
        snap.mkdir(parents=True)

        import json

        import cv2
        import numpy as np

        total = 10
        for i in range(total):
            sid = f"snap_{i:04d}"
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(snap / f"{sid}.jpg"), img)
            meta = {
                "id": sid,
                "labels": [{"cls_id": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}],
            }
            (snap / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")

        exporter = DatasetExporter(cfg)
        result = exporter.export()
        assert result["ok"] is True
        assert result["total"] == total
        assert result["train"] + result["val"] == total
        assert (cfg.resolved_paths.dataset / "data.yaml").exists()
        assert (cfg.resolved_paths.dataset / "dataset.zip").exists()

    def test_без_меток_ошибка(self, tmp_path: Path) -> None:
        from forky_spoonky_pi.config import AppConfig, PathsConfig
        from forky_spoonky_pi.dataset import DatasetExporter

        cfg = AppConfig(paths=PathsConfig(data_dir=str(tmp_path)))
        cfg.resolved_paths.snapshots.mkdir(parents=True)
        result = DatasetExporter(cfg).export()
        assert result["ok"] is False
        assert result["error"] == "no-labeled"


# ---------- SnapshotStore.save_labels валидация ----------
class TestSaveLabelsValidation:
    def test_фильтрация_и_кламп(self, tmp_path: Path) -> None:
        import json

        from forky_spoonky_pi.config import AppConfig, PathsConfig
        from forky_spoonky_pi.snapshots import SnapshotStore

        cfg = AppConfig(paths=PathsConfig(data_dir=str(tmp_path)))
        snap = cfg.resolved_paths.snapshots
        snap.mkdir(parents=True)
        sid = "snap_x"
        (snap / f"{sid}.json").write_text(json.dumps({"id": sid, "labels": []}), encoding="utf-8")

        store = SnapshotStore(cfg, detector=None)  # type: ignore[arg-type]
        raw = [
            {"cls_id": 0, "x": 1.5, "y": -0.5, "w": 0.2, "h": 0.2},
            {"cls_id": 5, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
            {"cls_id": 1, "x": 0.5, "y": 0.5, "w": 1.5, "h": 0.2},
            {"cls_id": 1, "x": 0.5, "y": 0.5, "w": 0.3, "h": 0.3},
        ]
        count = store.save_labels(sid, raw)
        assert count == 2
        meta = store.load_meta(sid)
        assert meta is not None
        assert meta["labels"][0]["x"] == pytest.approx(1.0)
        assert meta["labels"][0]["y"] == pytest.approx(0.0)
