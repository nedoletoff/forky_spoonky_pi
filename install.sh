#!/usr/bin/env bash
# Локальная установка окружения проекта (venv + зависимости).
# Идемпотентно: повторный запуск обновляет зависимости.
#
# Использование:
#   bash install.sh          # инференс на CPU (Pi/ноут): CPU-only torch + headless OpenCV
#   bash install.sh --train  # + CUDA-сборка torch/torchvision для дообучения (ноут с GPU)

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SOURCE_DIR}/.venv"
PIP="${VENV_DIR}/bin/pip"

MODE="pi"
if [[ "${1:-}" == "--train" ]]; then
  MODE="train"
fi

log() { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Создаю venv: ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

log "Обновляю pip/wheel…"
"${PIP}" install --upgrade pip wheel

if [[ "${MODE}" == "train" ]]; then
  # Обучение: ставим CUDA-сборку PyTorch (индекс под CUDA 12.4).
  # Если у тебя другая версия CUDA/драйвера — поправь --index-url по pytorch.org.
  log "Ставлю CUDA-сборку torch/torchvision…"
  "${PIP}" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  log "Ставлю зависимости обучения (requirements-train.txt)…"
  "${PIP}" install -r "${SOURCE_DIR}/requirements-train.txt"
else
  # Инференс: обязательно CPU-only torch, иначе pip тянет многогигабайтные CUDA-колёса.
  log "Ставлю CPU-only torch/torchvision (инференс)…"
  "${PIP}" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  log "Ставлю зависимости инференса (requirements-pi.txt)…"
  "${PIP}" install -r "${SOURCE_DIR}/requirements-pi.txt"
fi

log "Устанавливаю пакет проекта в editable-режиме…"
"${PIP}" install -e "${SOURCE_DIR}"

log "Готово. Запуск: bash run.sh  (или ./run.sh)"
