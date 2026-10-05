#!/usr/bin/env bash
# Локальная установка окружения проекта (venv + зависимости).
# Идемпотентно: повторный запуск обновляет зависимости.
#
# Использование:
#   bash install.sh          # окружение для инференса (Pi/ноут, CPU)
#   bash install.sh --train  # + torch/torchvision для дообучения (ноут с GPU)

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SOURCE_DIR}/.venv"

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
"${VENV_DIR}/bin/pip" install --upgrade pip wheel

if [[ "${MODE}" == "train" ]]; then
  log "Ставлю зависимости для обучения (requirements-train.txt)…"
  "${VENV_DIR}/bin/pip" install -r "${SOURCE_DIR}/requirements-train.txt"
else
  log "Ставлю зависимости инференса (requirements-pi.txt)…"
  "${VENV_DIR}/bin/pip" install -r "${SOURCE_DIR}/requirements-pi.txt"
fi

log "Готово. Запуск: bash run.sh  (или ./run.sh)"
