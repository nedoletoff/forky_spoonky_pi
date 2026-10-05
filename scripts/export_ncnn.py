#!/usr/bin/env python3
"""Экспорт обученной модели в NCNN для Raspberry Pi.

Запускать на ноутбуке (после дообучения). NCNN на ARM существенно быстрее
PyTorch: на Raspberry Pi 5 порядка 60–70 мс/кадр против ~300 мс.

Использование:
    python scripts/export_ncnn.py data/runs/train_.../weights/best.pt
    python scripts/export_ncnn.py best.pt --imgsz 416 --out data/weights

Результат — каталог best_ncnn_model/ рядом с .pt (или в --out). На Pi укажи его
в .env:  FORKY_MODEL__WEIGHTS=/opt/forky_spoonky_pi/data/weights/best_ncnn_model
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Экспорт YOLO-модели в NCNN")
    parser.add_argument("weights", help="Путь к .pt-файлу (например, best.pt)")
    parser.add_argument("--imgsz", type=int, default=416, help="Размер входа (по умолчанию 416)")
    parser.add_argument("--out", default=None, help="Каталог для готовой NCNN-модели")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        print(f"Файл не найден: {weights}", file=sys.stderr)
        return 1

    print(f"Экспортирую {weights} в NCNN (imgsz={args.imgsz})…")
    model = YOLO(str(weights))
    exported = Path(model.export(format="ncnn", imgsz=args.imgsz))
    print(f"Готово: {exported}")

    if args.out:
        dest = Path(args.out).resolve() / exported.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(exported, dest)
        print(f"Скопировано в: {dest}")
        exported = dest

    print("\nСкопируй каталог на Raspberry Pi и укажи в .env:")
    print(f"  FORKY_MODEL__WEIGHTS={exported}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
