# forky_spoonky_pi

Детектор столовых приборов (**вилки** и **ложки**) на YOLO с веб-интерфейсом: живой поток с камеры, снимки, разметка боксов в браузере, экспорт YOLO-датасета, экспорт фотографий (ZIP) и дообучение модели прямо из интерфейса с прогрессом, ETA и графиком метрик.

Проект рассчитан на Raspberry Pi (`/opt/forky_spoonky_pi`) для инференса, а дообучение — на ноутбуке с дискретной видеокартой (на Pi обучение отключено).

## Возможности

- **Live-поток** MJPEG с камеры, счётчики вилок/ложек/непонятных объектов, FPS и время инференса.
- **Снимки** — сохранение кадра, авто-разметка детекцией, ручной редактор боксов (canvas).
- **Переобработка** всех снимков при смене порога уверенности (ручная разметка не затирается).
- **Экспорт датасета** в формате YOLO (`images/labels/{train,val}`, `data.yaml`, `dataset.zip`).
- **Экспорт фотографий** — ZIP всех снимков (чистых или с нарисованными боксами) и скачивание отдельного кадра.
- **Дообучение** из веб-интерфейса: выбор базовой модели, эпох, размера, батча; оверлей поверх
  зоны камеры с прогрессом, ETA и графиком метрик (Chart.js, локально завендорен).
- **Экспорт модели** в NCNN/ONNX и **promote** — переключение рабочей модели без перезапуска.
- **Конфигурация** из `config.yaml` с переопределением через переменные окружения `FORKY_*` и `.env`.
- **Логи** с ротацией в `data/logs/app.log` и выводом в stdout (journald при работе через systemd).

## Структура

```
.
├── config.yaml                 # основной конфиг
├── .env.example                # пример переменных окружения (копируется в .env)
├── requirements-pi.txt         # инференс (Pi/ноут, CPU)
├── requirements-train.txt      # + torch/torchvision для дообучения (ноут с GPU)
├── install.sh                  # создание venv и установка зависимостей
├── run.sh                      # запуск приложения
├── deploy/
│   ├── forky_spoonky_pi.service  # systemd-юнит
│   └── install-pi.sh             # установка/обновление на Pi в /opt
├── scripts/
│   └── export_ncnn.py          # best.pt -> NCNN (запуск на ноуте)
├── src/forky_spoonky_pi/       # пакет приложения
│   ├── __main__.py             # точка входа: python -m forky_spoonky_pi
│   ├── config.py               # загрузка конфига + env-оверрайды
│   ├── logging_setup.py        # настройка логирования с ротацией
│   ├── state.py                # разделяемое состояние (поток камеры/HTTP)
│   ├── model.py                # загрузка/подмена модели
│   ├── camera.py               # воркер камеры
│   ├── detector.py             # инференс и отрисовка
│   ├── snapshots.py            # снимки и разметка
│   ├── dataset.py              # экспорт датасета
│   ├── exporter.py             # экспорт фотографий (ZIP/кадр)
│   ├── training.py             # запуск обучения, метрики, ETA
│   └── web/                    # Flask: routes, templates, static
└── data/                       # создаётся автоматически (см. .gitignore)
    ├── snapshots/  dataset/  runs/  logs/  weights/
```

## Быстрый старт (локально)

```bash
bash install.sh          # creates .venv, CPU-only torch, deps, editable install
cp .env.example .env     # tweak if needed
./run.sh
```

Open `http://<host>:8080`. The `yolo26n.pt` model downloads automatically on first start.

> **torch и место на диске.** `install.sh` намеренно ставит CPU-only сборку torch
> (`--index-url .../whl/cpu`, ~200 МБ). Если поставить `torch` обычным `pip install`,
> pip притянет CUDA-колёса на несколько гигабайт — на машине без GPU это не нужно
> и часто не проходит по месту в кэше pip (`pip cache purge` освобождает место).

## Конфигурация

Все параметры — в `config.yaml`. Любое значение переопределяется переменной окружения вида `FORKY_<СЕКЦИЯ>__<КЛЮЧ>` (двойное подчёркивание — разделитель уровней), удобно хранить в `.env`:

```bash
FORKY_SERVER__PORT=9000
FORKY_CAMERA__INDEX=1
FORKY_MODEL__WEIGHTS=/opt/forky_spoonky_pi/data/weights/best_ncnn_model
FORKY_LOGGING__LEVEL=DEBUG
```

Секреты (если появятся) держим только в `.env` — он в `.gitignore`.

## Камера: USB или CSI

Тип источника задаётся в `config.yaml` → `camera.source`:

| Значение | Что это | Как работает |
|----------|---------|--------------|
| `usb`    | USB-вебка или CSI-камера в режиме V4L2 | OpenCV `cv2.VideoCapture` по `camera.index` или `camera.device` |
| `csi`    | Камера Raspberry Pi (шлейф CSI) через libcamera | `picamera2` |

Параметры:

- `camera.index` — индекс V4L2-устройства (для `source: usb`), по умолчанию `0` → `/dev/video0`.
- `camera.device` — явный путь, например `/dev/video0`; если задан, имеет приоритет над `index`.
- `camera.width` / `camera.height` / `camera.fps` — разрешение и частота.
- `camera.use_v4l2` — использовать backend V4L2 (для `source: usb`).

### USB-камера

```yaml
camera:
  source: usb
  index: 0
```

Проверь, что устройство видно: `ls /dev/video*`. Пользователю, от которого запущен сервис, нужен доступ к камере — в systemd-юните уже указан `SupplementaryGroups=video`.

### CSI-камера Raspberry Pi

```yaml
camera:
  source: csi
  width: 640
  height: 480
```

Порядок настройки на Pi:

1. Включи камеру: `sudo raspi-config` → Interface Options → Camera (или `camera_auto_detect=1` в `/boot/firmware/config.txt`), перезагрузись.
2. Проверь камеру: `rpicam-hello --list-cameras`.
3. Установи `picamera2` (идёт из apt, а не из pip):
   ```bash
   sudo apt install -y python3-picamera2 libcamera-apps
   ```
4. При установке через `deploy/install-pi.sh` поддержка CSI ставится флагом: `sudo WITH_CSI=1 bash deploy/install-pi.sh`. Скрипт создаёт venv с `--system-site-packages`, чтобы `picamera2` из apt был виден.
5. Перезапусти сервис: `sudo systemctl restart forky_spoonky_pi`.

Если `picamera2` не установлен, приложение не падает: в лог пишется понятная ошибка, а в интерфейсе показывается «Камера недоступна».

## Дообучение и перенос на Raspberry Pi

Обучение на Pi нецелесообразно (нет GPU). Рекомендуемый цикл:

1. **На ноутбуке** установи зависимости с обучением: `bash install.sh --train`  (при необходимости сначала поставь CUDA-совместимый PyTorch — см. `requirements-train.txt`).
2. Включи обучение в конфиге: в `config.yaml` → `training.enabled: true` (или `FORKY_TRAINING__ENABLED=true`).
3. Собери данные: наснимай кадры, разметь боксы в браузере, экспортируй датасет (кнопка «Экспорт датасета»).
4. Запусти обучение в веб-интерфейсе. Прогресс, ETA и метрики появятся в оверлее поверх камеры. Результат — `data/runs/<name>/weights/best.pt`.
5. **Экспортируй в NCNN** (на ARM существенно быстрее PyTorch): 
   ```bash
   python scripts/export_ncnn.py data/runs/<name>/weights/best.pt --out data/weights
   ```
6. Скопируй `data/weights/best_ncnn_model` на Pi и укажи путь в `.env`:
   ```bash
   FORKY_MODEL__WEIGHTS=/opt/forky_spoonky_pi/data/weights/best_ncnn_model
   ```
7. Перезапусти сервис. Альтернатива без правки `.env` — кнопки «Promote» (переключить рабочую модель) и «Экспорт NCNN» прямо в интерфейсе.

> Примечание: для инференса NCNN на Pi может потребоваться пакет `python3-ncnn` или установка `ncnn` из исходников; ultralytics умеет работать с NCNN-каталогом напрямую.

## Установка на Raspberry Pi (systemd)

Скопируй папку проекта на Pi и запусти установщик:

```bash
sudo bash deploy/install-pi.sh
```

Скрипт идемпотентен: ставит системные пакеты (`libgl1`, `libglib2.0-0`, `rsync`, `python3-venv`), синхронизирует код в `/opt/forky_spoonky_pi` (не трогая `data/` и `.env`), создаёт venv, ставит `requirements-pi.txt`, копирует `.env.example` → `.env`, включает и запускает systemd-сервис. Для CSI-камеры добавь `WITH_CSI=1` — скрипт доложит `python3-picamera2` и `libcamera-apps`.

Полезные команды:

```bash
systemctl status forky_spoonky_pi
journalctl -u forky_spoonky_pi -f
tail -f /opt/forky_spoonky_pi/data/logs/app.log
```

Пользователя сервиса (`User=pi`) и путь установки можно поправить в `deploy/forky_spoonky_pi.service` и `deploy/install-pi.sh`.

## Разработка

```bash
.venv/bin/ruff check src tests scripts   # линтер
.venv/bin/python -m pytest               # тесты
```

## Лицензия

MIT.
