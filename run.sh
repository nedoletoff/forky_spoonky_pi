#!/usr/bin/env bash
# Запуск приложения из venv проекта.
#
# Использование:
#   ./run.sh              # обычный запуск
#   CONFIG=/path/config.yaml ./run.sh
#   FORKY_SERVER__PORT=9000 ./run.sh

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${SOURCE_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Не найден venv: ${VENV_PY}. Сначала выполни: bash install.sh" >&2
  exit 1
fi

cd "${SOURCE_DIR}"
exec "${VENV_PY}" -m forky_spoonky_pi "$@"
