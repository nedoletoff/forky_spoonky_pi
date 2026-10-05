"""HTTP-маршруты приложения (единый blueprint)."""

from __future__ import annotations

import io
import time

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)

from .. import state as state_module
from ..config import AppConfig
from ..dataset import DatasetExporter
from ..exporter import PhotoExporter
from ..snapshots import SnapshotStore
from ..training import Trainer

bp = Blueprint("forky", __name__)


def _ctx():
    config: AppConfig = current_app.config["FORKY"]
    detector = current_app.extensions["forky_detector"]
    holder = current_app.extensions["forky_holder"]
    store = SnapshotStore(config, detector)
    dataset = DatasetExporter(config)
    photos = PhotoExporter(config, detector)
    trainer = Trainer(config)
    return config, detector, holder, store, dataset, photos, trainer


# ================== Страница и поток ==================
@bp.route("/")
def index():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    return render_template(
        "index.html",
        classes=config.classes.names_ru,
        model_name=holder.display_name,
        threshold=config.model.confidence,
        imgsz=config.model.imgsz,
        port=config.server.port,
        training_enabled=config.training.enabled,
    )


def _generate_frames():
    last = None
    while True:
        with state_module.lock:
            frame = state_module.state["latest_jpeg"]
        if frame is None or frame is last:
            time.sleep(0.02)
            continue
        last = frame
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@bp.route("/video_feed")
def video_feed():
    return Response(_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/stats")
def stats():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    with state_module.lock:
        payload = dict(state_module.state["stats"])
    payload["model"] = holder.display_name
    return jsonify(payload)


# ================== Снимки ==================
@bp.route("/snapshot", methods=["POST"])
def snapshot():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    result = store.capture()
    if result is None:
        return jsonify({"ok": False, "error": "no-frame"}), 400
    return jsonify({"ok": True, **result})


@bp.route("/snapshots")
def snapshots_list():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    return jsonify(store.list())


@bp.route("/snapshots/reprocess-all", methods=["POST"])
def snap_reprocess_all():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    threshold = float(data.get("threshold", config.model.confidence))
    processed = store.reprocess_all(threshold)
    return jsonify({"ok": True, "processed": processed})


@bp.route("/snapshots/<sid>")
def snap_meta(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    meta = store.load_meta(sid)
    if meta is None:
        abort(404)
    return jsonify(meta)


@bp.route("/snapshots/<sid>/image")
def snap_image(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    p = store.image_path(sid)
    if p is None:
        abort(404)
    return send_file(p, mimetype="image/jpeg")


@bp.route("/snapshots/<sid>/thumb")
def snap_thumb(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    p = store.image_path(sid)
    if p is None:
        abort(404)
    import cv2

    img = cv2.imread(str(p))
    if img is None:
        abort(404)
    meta = store.load_meta(sid)
    if meta and meta.get("labels"):
        img = detector.draw_yolo_labels(img, meta["labels"])
    h, w = img.shape[:2]
    scale = 320 / w
    if scale < 1:
        img = cv2.resize(img, (320, int(h * scale)), interpolation=cv2.INTER_AREA)
    buf = detector.encode_jpeg(img, config.camera.jpeg_quality)
    if not buf:
        abort(500)
    return Response(buf, mimetype="image/jpeg")


@bp.route("/snapshots/<sid>", methods=["DELETE"])
def snap_delete(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    return jsonify({"ok": store.delete(sid)})


@bp.route("/snapshots/<sid>/labels", methods=["POST"])
def snap_save_labels(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    count = store.save_labels(sid, data.get("labels", []))
    if count is None:
        return jsonify({"ok": False, "error": "not-found"}), 404
    return jsonify({"ok": True, "count": count})


@bp.route("/snapshots/<sid>/detect", methods=["POST"])
def snap_detect(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    threshold = float(data.get("threshold", config.model.confidence))
    result = store.detect(sid, threshold)
    if result is None:
        return jsonify({"ok": False, "error": "not-found"}), 404
    return jsonify({"ok": True, **result})


@bp.route("/snapshots/<sid>/load", methods=["POST"])
def snap_load(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    threshold = float(data.get("threshold", config.model.confidence))
    result = store.load_to_view(sid, threshold)
    if result is None:
        return jsonify({"ok": False, "error": "not-found"}), 404
    return jsonify({"ok": True, **result})


# ================== Экспорт фото ==================
@bp.route("/photos/zip")
def photos_zip():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    annotated = request.args.get("annotated", "0") in ("1", "true", "yes")
    data = photos.zip_all(annotated)
    if data is None:
        abort(404)
    suffix = "annotated" if annotated else "clean"
    return send_file(
        io.BytesIO(data),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"photos_{suffix}.zip",
    )


@bp.route("/snapshots/<sid>/download")
def snap_download(sid):
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    annotated = request.args.get("annotated", "0") in ("1", "true", "yes")
    data = photos.single_frame(sid, annotated)
    if data is None:
        abort(404)
    suffix = "_annotated" if annotated else ""
    return send_file(
        io.BytesIO(data),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name=f"{sid}{suffix}.jpg",
    )


# ================== Параметры и поток ==================
@bp.route("/config", methods=["POST"])
def config_route():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    with state_module.lock:
        if "threshold" in data:
            t = max(0.05, min(0.95, float(data["threshold"])))
            state_module.state["stats"]["threshold"] = t
        threshold = state_module.state["stats"]["threshold"]
    return jsonify({"ok": True, "threshold": threshold})


@bp.route("/resume", methods=["POST"])
def resume():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    with state_module.lock:
        state_module.state["mode"] = "live"
        state_module.state["stats"]["mode"] = "live"
        state_module.state["frozen_bgr"] = None
    return jsonify({"ok": True, "mode": "live"})


@bp.route("/redetect", methods=["POST"])
def redetect():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    with state_module.lock:
        if state_module.state["mode"] != "frozen" or state_module.state["frozen_bgr"] is None:
            return jsonify({"ok": False, "error": "not-frozen"}), 400
        raw = state_module.state["frozen_bgr"].copy()
        default_thr = state_module.state["stats"]["threshold"]
        status = state_module.state["stats"].get("status", "online")

    data = request.get_json(silent=True) or {}
    threshold = max(0.05, min(0.95, float(data.get("threshold", default_thr))))
    drawn, counts, infer_ms, dets = detector.process_frame(raw, threshold)
    detector.overlay_meta(drawn, infer_ms=infer_ms, frozen=True)
    buf = detector.encode_jpeg(drawn, config.camera.jpeg_quality)

    with state_module.lock:
        state_module.state["display_bgr"] = drawn
        state_module.state["last_detections"] = dets
        if buf:
            state_module.state["latest_jpeg"] = buf
        state_module.state["stats"] = {
            **counts,
            "fps": state_module.state["stats"].get("fps", 0.0),
            "infer_ms": round(infer_ms, 0),
            "status": status,
            "mode": "frozen",
            "threshold": threshold,
        }
    return jsonify({"ok": True, "counts": counts, "infer_ms": round(infer_ms, 0)})


@bp.route("/download")
def download():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    with state_module.lock:
        if state_module.state["display_bgr"] is None:
            return ("No frame", 404)
        drawn = state_module.state["display_bgr"].copy()
        frozen = state_module.state["mode"] == "frozen"
    buf = detector.encode_jpeg(drawn, quality=95)
    if not buf:
        return ("Encode failed", 500)
    name = ("snapshot_" if frozen else "frame_") + str(int(time.time())) + ".jpg"
    return send_file(io.BytesIO(buf), mimetype="image/jpeg", as_attachment=True, download_name=name)


# ================== Датасет ==================
@bp.route("/dataset/export", methods=["POST"])
def dataset_export():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    result = dataset.export()
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@bp.route("/dataset/download")
def dataset_download():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    p = config.resolved_paths.dataset / "dataset.zip"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="application/zip", as_attachment=True, download_name="dataset.zip")


# ================== Обучение ==================
@bp.route("/train", methods=["POST"])
def train_start():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    result = trainer.start(data)
    if not result.get("ok"):
        code = 409 if result.get("error") == "already-running" else 400
        return jsonify(result), code
    return jsonify(result)


@bp.route("/train/stop", methods=["POST"])
def train_stop():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    return jsonify({"ok": trainer.stop()})


@bp.route("/train/status")
def train_status():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    return jsonify(trainer.status())


@bp.route("/train/metrics")
def train_metrics():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    return jsonify(trainer.metrics())


@bp.route("/model/promote", methods=["POST"])
def model_promote():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    path = data.get("path")
    if not path:
        return jsonify({"ok": False, "error": "bad-path"}), 400
    with state_module.model_lock:
        ok = holder.promote(path)
    if not ok:
        return jsonify({"ok": False, "error": "load-failed"}), 500
    return jsonify({"ok": True, "model": path})


@bp.route("/model/export", methods=["POST"])
def model_export():
    config, detector, holder, store, dataset, photos, trainer = _ctx()
    data = request.get_json(silent=True) or {}
    path = data.get("path") or holder.weights
    fmt = data.get("format", "ncnn")
    out = detector.export_model(path, fmt)
    if out is None:
        return jsonify({"ok": False, "error": "export-failed"}), 500
    return jsonify({"ok": True, "path": out})
