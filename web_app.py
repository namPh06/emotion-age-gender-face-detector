"""Flask web app for face age, gender, and emotion detection."""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from src.detect_face import FaceDetectionConfig, crop_face_bgr
from src.pipeline import analyze_frame_bgr, load_models


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

_models: object | None = None
_models_error: str | None = None
_loader_thread: threading.Thread | None = None
_loader_lock = threading.Lock()
_inference_lock = threading.Lock()


def _load_models_worker() -> None:
    global _models, _models_error
    try:
        logger.info("Loading models for web app")
        _models = load_models(PROJECT_DIR / "models")
        _models_error = None
        logger.info("Models ready")
    except Exception as exc:
        logger.exception("Could not load models")
        _models_error = str(exc)


def start_model_loading() -> None:
    """Start the model loader thread once."""
    global _loader_thread
    if _models is not None:
        return
    with _loader_lock:
        if _models is not None:
            return
        if _loader_thread is not None and _loader_thread.is_alive():
            return
        _loader_thread = threading.Thread(target=_load_models_worker, daemon=True)
        _loader_thread.start()


def _model_status() -> dict[str, Any]:
    if _models is not None:
        return {"state": "ready", "message": "Ready"}
    if _models_error is not None:
        return {"state": "error", "message": _models_error}
    if _loader_thread is not None and _loader_thread.is_alive():
        return {"state": "loading", "message": "Loading models"}
    return {"state": "idle", "message": "Models not loaded"}


def _decode_image_from_request() -> np.ndarray:
    upload = request.files.get("image")
    if upload is not None:
        data = upload.read()
    else:
        payload = request.get_json(silent=True) or {}
        image_data = str(payload.get("image", ""))
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        data = base64.b64decode(image_data)

    image = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data.")
    return frame


def _request_value(name: str, default: str) -> str:
    if name in request.form:
        return str(request.form.get(name) or default)
    payload = request.get_json(silent=True) or {}
    return str(payload.get(name, default))


def _make_config() -> FaceDetectionConfig:
    detector = _request_value("detector", "yolo").lower()
    quality = _request_value("quality", "fast").lower()
    source = _request_value("source", "webcam").lower()

    if detector not in {"yolo", "auto", "haar"}:
        detector = "yolo"

    if source == "image":
        yolo_imgsz = 640 if quality != "fast" else 512
        min_size = (24, 24)
        margin = 0.35
    else:
        yolo_imgsz = 416 if quality == "fast" else 640
        min_size = (32, 32)
        margin = 0.30

    return FaceDetectionConfig(
        backend=detector,
        min_neighbors=4,
        min_size=min_size,
        margin=margin,
        yolo_imgsz=yolo_imgsz,
        yolo_confidence=0.32 if quality == "fast" else 0.28,
        yolo_iou=0.45,
        yolo_max_det=5 if source == "webcam" else 20,
    )


def _should_return_image() -> bool:
    value = _request_value("return_image", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _prediction_payload(
    predictions,
    original_frame_bgr: np.ndarray,
    config: FaceDetectionConfig,
    *,
    include_crops: bool = True,
) -> list[dict[str, Any]]:
    payload = []
    for prediction in predictions:
        x, y, w, h = prediction.box
        result = prediction.result
        crop_image = None
        if include_crops:
            try:
                crop = crop_face_bgr(original_frame_bgr, prediction.box, margin=config.margin)
                crop_image = _encode_jpeg_data_url(crop, quality=84)
            except Exception as exc:
                logger.warning("Could not encode face crop: %s", exc)

        payload.append(
            {
                "box": {"x": x, "y": y, "width": w, "height": h},
                "crop_image": crop_image,
                "gender": result.gender,
                "gender_confidence": round(result.gender_confidence, 4),
                "age": result.age,
                "age_confidence": round(result.age_confidence, 4),
                "emotion": result.emotion,
                "emotion_confidence": round(result.emotion_confidence, 4),
            }
        )
    return payload


def _encode_jpeg_data_url(frame_bgr: np.ndarray, *, quality: int = 88) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Could not encode annotated image.")
    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    start_model_loading()
    yolo_path = PROJECT_DIR / "models" / "face_detection" / "yolo_face.pt"
    return jsonify(
        {
            "models": _model_status(),
            "detector": {
                "default": "yolo",
                "yolo_model_exists": yolo_path.exists(),
                "yolo_model_path": str(yolo_path),
            },
        }
    )


@app.post("/api/analyze")
def analyze():
    if _models is None:
        start_model_loading()
        return jsonify({"error": "Models are still loading.", "models": _model_status()}), 503

    started = time.perf_counter()
    try:
        frame = _decode_image_from_request()
        original_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        config = _make_config()
        return_image = _should_return_image()
        max_faces = config.yolo_max_det
        with _inference_lock:
            predictions = analyze_frame_bgr(
                frame,
                _models,
                config,
                max_faces=max_faces,
                annotate=return_image,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        payload = {
            "image": _encode_jpeg_data_url(frame) if return_image else None,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "face_count": len(predictions),
            "elapsed_ms": elapsed_ms,
            "predictions": _prediction_payload(
                predictions,
                original_frame,
                config,
                include_crops=return_image,
            ),
        }
        return jsonify(payload)
    except Exception as exc:
        logger.exception("Analyze request failed")
        return jsonify({"error": str(exc)}), 400


def main() -> None:
    start_model_loading()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
