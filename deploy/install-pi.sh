#!/usr/bin/env bash
# Установка/обновление приложения на Raspberry Pi в /opt/forky_spoonky_pi.
# Идемпотентный скрипт: повторный запуск обновляет код, не затирая .env и data/.
#
# Использование (на Pi, из распакованной папки проекта):
#   sudo bash deploy/install-pi.sh
#
# Перед запуском положи рядом (в корень проекта) готовую NCNN-модель, если она
# уже обучена, например data/weights/best_ncnn_model/, и укажи путь в .env:
#   FORKY_MODEL__WEIGHTS=/opt/forky_spoonky_pi/data/weights/best_ncnn_model

set -euo pipefail

APP_NAME="forky_spoonky_pi"
APP_DIR="/opt/${APP_NAME}"
SERVICE_NAME="${APP_NAME}.service"
SERVICE_USER="pi"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

if [[ "${EUID}" -ne 0 ]]; then
  err "Запусти скрипт от root: sudo bash deploy/install-pi.sh"
  exit 1
fi

if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
  err "Пользователь ${SERVICE_USER} не найден. Поправь SERVICE_USER в скрипте."
  exit 1
fi

log "Источник: ${SOURCE_DIR}"
log "Назначение: ${APP_DIR}"

# 1. Устанавливаем системные зависимости (libGL нужен для opencv, хотя headless
#    обычно обходится, ставим на всякий случай для Pi 3/4/5).
log "Устанавливаю системные пакеты (libgl1, libglib2.0-0, rsync, python3-venv)…"
apt-get update -y
apt-get install -y --no-install-recommends \
  python3-venv python3-pip rsync \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

# 2. Копируем код (кроме данных, venv и git-мусора).
log "Синхронизирую файлы проекта в ${APP_DIR}…"
mkdir -p "${APP_DIR}"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'data' \
  --exclude '.env' \
  "${SOURCE_DIR}/" "${APP_DIR}/"

# 3. venv и зависимости (инференс).
log "Создаю venv и ставлю зависимости…"
if [[ ! -d "${APP_DIR}/.venv" ]]; then
  python3 -m venv "${APP_DIR}/.venv"
fi
"${APP_DIR}/.venv/bin/pip" install --upgrade pip wheel
# CPU-only torch ОБЯЗАТЕЛЬНО до ultralytics: иначе pip тянет CUDA-колёса на гигабайты,
# что не нужно для инференса на Pi (и часто не проходит по месту/сети).
log "Ставлю CPU-only torch/torchvision…"
"${APP_DIR}/.venv/bin/pip" install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements-pi.txt"
"${APP_DIR}/.venv/bin/pip" install -e "${APP_DIR}"

# 4. .env из примера (не перезаписываем существующий — там могут быть секреты).
if [[ ! -f "${APP_DIR}/.env" ]]; then
  log "Создаю ${APP_DIR}/.env из .env.example"
  cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
else
  log ".env уже существует — оставляю без изменений"
fi

# 5. Права на данные и код.
log "Выставляю владельца ${SERVICE_USER} на ${APP_DIR}…"
mkdir -p "${APP_DIR}/data"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"
chmod +x "${APP_DIR}/run.sh" 2>/dev/null || true

# 6. systemd.
log "Устанавливаю systemd-юнит…"
install -m 0644 "${APP_DIR}/deploy/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

log "Готово. Статус:"
systemctl --no-pager --full status "${SERVICE_NAME}" || true
log "Логи: journalctl -u ${SERVICE_NAME} -f  или  tail -f ${APP_DIR}/data/logs/app.log"
