"""Face detection, cropping, and annotation utilities.

YOLO-face is the preferred detector for better accuracy on small,
rotated, and poorly-lit faces. Haar Cascade remains available as a
lightweight fallback so the app can still run without the YOLO package
or model file.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_MODEL_PATH = PROJECT_DIR / "models" / "face_detection" / "yolo_face.pt"
YOLO_CONFIG_DIR = PROJECT_DIR / ".ultralytics"
SUPPORTED_BACKENDS = {"auto", "yolo", "haar"}


@dataclass(frozen=True)
class FaceDetectionConfig:
    """Tunable parameters for face detection.

    Attributes
    ----------
    scale_factor:
        How much the image size is reduced at each image scale.
    min_neighbors:
        How many neighbours each candidate rectangle should have
        to retain it.
    min_size:
        Minimum possible face size (width, height).
    margin:
        Fractional padding added around detected face boxes before
        cropping.
    backend:
        Detector backend: ``"yolo"``, ``"haar"``, or ``"auto"``.
    yolo_model_path:
        Path to a YOLO face ``.pt`` model. If omitted, the project
        default in ``models/face_detection/yolo_face.pt`` is used.
    yolo_confidence:
        Minimum confidence for YOLO face detections.
    yolo_iou:
        IoU threshold for YOLO non-maximum suppression.
    yolo_imgsz:
        Inference image size passed to YOLO.
    """
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (48, 48)
    margin: float = 0.35
    backend: str = "auto"
    yolo_model_path: str | None = None
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    yolo_max_det: int = 20


def _load_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Cannot load OpenCV Haar cascade: {cascade_path}")
    return cascade


FACE_CASCADE = _load_cascade()
_DETECTOR_CACHE: dict[tuple[object, ...], object] = {}


class HaarFaceDetector:
    """OpenCV Haar Cascade face detector."""

    def detect(self, frame_bgr: np.ndarray, config: FaceDetectionConfig) -> List[Box]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=config.scale_factor,
            minNeighbors=config.min_neighbors,
            minSize=config.min_size,
        )
        boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        return _sort_boxes(boxes)


class YoloFaceDetector:
    """Ultralytics YOLO face detector."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO face model not found: {model_path}. "
                "Run scripts/download_yolo_face.py or place a YOLO-face .pt file there."
            )

        YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'ultralytics'. Install requirements.txt first."
            ) from exc

        self.model_path = model_path
        self.model = YOLO(str(model_path))

    def detect(self, frame_bgr: np.ndarray, config: FaceDetectionConfig) -> List[Box]:
        results = self.model.predict(
            frame_bgr,
            imgsz=int(config.yolo_imgsz),
            conf=float(config.yolo_confidence),
            iou=float(config.yolo_iou),
            max_det=int(config.yolo_max_det),
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        boxes: List[Box] = []
        for xyxy in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy[:4]]
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width >= config.min_size[0] and height >= config.min_size[1]:
                boxes.append((x1, y1, width, height))

        return _sort_boxes(boxes)


def _normalise_backend(backend: str) -> str:
    backend = backend.strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        logger.warning("Unknown face detector backend '%s'; using auto.", backend)
        return "auto"
    return backend


def _resolve_yolo_model_path(config: FaceDetectionConfig) -> Path:
    if config.yolo_model_path:
        return Path(config.yolo_model_path)
    return DEFAULT_YOLO_MODEL_PATH


def _detector_cache_key(config: FaceDetectionConfig) -> tuple[object, ...]:
    backend = _normalise_backend(config.backend)
    if backend == "haar":
        return ("haar",)
    return (
        backend,
        str(_resolve_yolo_model_path(config).resolve()),
    )


def _create_detector(config: FaceDetectionConfig) -> object:
    backend = _normalise_backend(config.backend)
    if backend == "haar":
        return HaarFaceDetector()

    try:
        return YoloFaceDetector(_resolve_yolo_model_path(config))
    except Exception as exc:
        logger.warning("Could not load YOLO face detector; falling back to Haar: %s", exc)
        return HaarFaceDetector()


def _get_detector(config: FaceDetectionConfig) -> object:
    key = _detector_cache_key(config)
    detector = _DETECTOR_CACHE.get(key)
    if detector is None:
        detector = _create_detector(config)
        _DETECTOR_CACHE[key] = detector
    return detector


def _sort_boxes(boxes: List[Box]) -> List[Box]:
    return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)


def detect_faces_bgr(
    frame_bgr: np.ndarray,
    config: FaceDetectionConfig | None = None,
) -> List[Box]:
    """Detect faces in a BGR image and return boxes as ``(x, y, w, h)``.

    Boxes are sorted by area in descending order (largest face first).
    """
    if config is None:
        config = FaceDetectionConfig()

    detector = _get_detector(config)
    return detector.detect(frame_bgr, config)


def expand_box(box: Box, image_shape: Tuple[int, ...], margin: float = 0.35) -> Box:
    """Expand a detection box by *margin* while clamping to image bounds."""
    x, y, w, h = box
    height, width = image_shape[:2]
    pad = int(max(w, h) * margin)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def crop_face_bgr(
    frame_bgr: np.ndarray,
    box: Box,
    margin: float = 0.35,
) -> np.ndarray:
    """Crop a face region from a BGR frame with padding.

    Raises
    ------
    ValueError
        If the resulting crop has zero pixels (e.g. box at the very
        edge of the frame).
    """
    x, y, w, h = expand_box(box, frame_bgr.shape, margin)
    face = frame_bgr[y : y + h, x : x + w]
    if face.size == 0:
        raise ValueError(
            f"Cropped face is empty: original_box={box}, "
            f"expanded=({x}, {y}, {w}, {h}), "
            f"frame_shape={frame_bgr.shape}"
        )
    return face


def _measure_label_lines(
    lines: list[str],
    font: int,
    font_scale: float,
    thickness: int,
    line_gap: int,
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Measure text metrics for a multi-line label."""
    metrics: list[tuple[int, int, int]] = []
    max_width = 0
    total_height = 0

    for index, line in enumerate(lines):
        (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        metrics.append((text_w, text_h, baseline))
        max_width = max(max_width, text_w)
        total_height += text_h + baseline
        if index < len(lines) - 1:
            total_height += line_gap

    return metrics, max_width, total_height


def _choose_label_layout(
    frame_shape: tuple[int, ...],
    gender_label: str,
    age_label: str,
    emotion_label: str,
    score: float | None,
) -> tuple[list[str], float, list[tuple[int, int, int]], int, int, int]:
    """Pick the most readable label layout that fits inside the image."""
    frame_h, frame_w = frame_shape[:2]
    max_width = max(1, frame_w - 12)
    max_height = max(1, frame_h - 12)
    # Confidence values are shown in the web result cards. Keep the image
    # annotation focused on the final labels only.
    _ = score

    # Prefer a single line when it fits, but fall back to shorter multi-line
    # layouts for small images so the annotation is not clipped.
    label_variants: list[list[str]] = [
        [f"{gender_label} | {age_label} | {emotion_label}"],
        [f"{gender_label} | {age_label}", emotion_label],
        [gender_label, age_label, emotion_label],
    ]

    preferred_scale = 0.55
    min_scale = 0.30
    best_choice: tuple[float, int, list[str], list[tuple[int, int, int]], int, int, int] | None = None

    for variant_index, lines in enumerate(label_variants):
        scale = preferred_scale
        while scale >= min_scale:
            line_gap = max(2, int(round(scale * 4)))
            metrics, text_w, text_h = _measure_label_lines(lines, cv2.FONT_HERSHEY_SIMPLEX, scale, 2, line_gap)
            if text_w <= max_width and text_h <= max_height:
                choice = (scale, variant_index, lines, metrics, text_w, text_h, line_gap)
                if best_choice is None:
                    best_choice = choice
                else:
                    best_scale, _, best_lines, *_ = best_choice
                    if scale > best_scale or (
                        abs(scale - best_scale) <= 1e-6 and len(lines) < len(best_lines)
                    ):
                        best_choice = choice
                break
            scale -= 0.05

    if best_choice is None:
        lines = label_variants[-1]
        scale = min_scale
        line_gap = max(2, int(round(scale * 4)))
        metrics, text_w, text_h = _measure_label_lines(lines, cv2.FONT_HERSHEY_SIMPLEX, scale, 2, line_gap)
        return lines, scale, metrics, text_w, text_h, line_gap

    _, _, lines, metrics, text_w, text_h, line_gap = best_choice
    return lines, best_choice[0], metrics, text_w, text_h, line_gap


def draw_prediction(
    frame_bgr: np.ndarray,
    box: Box,
    age_label: str,
    gender_label: str,
    emotion_label: str,
    score: float | None = None,
) -> None:
    """Draw a bounding box and prediction label on *frame_bgr* in-place."""
    x, y, w, h = box
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 220, 0), 2)

    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines, font_scale, metrics, text_w, text_h, line_gap = _choose_label_layout(
        frame_bgr.shape,
        gender_label,
        age_label,
        emotion_label,
        score,
    )

    padding_x = 6
    padding_y = 5
    panel_w = text_w + padding_x * 2
    panel_h = text_h + padding_y * 2
    frame_h, frame_w = frame_bgr.shape[:2]
    label_x = max(0, min(x, frame_w - panel_w))
    label_y = max(0, min(y - panel_h - 8, frame_h - panel_h))

    cv2.rectangle(
        frame_bgr,
        (label_x, label_y),
        (label_x + panel_w, label_y + panel_h),
        (0, 0, 0),
        -1,
    )

    cursor_y = label_y + padding_y
    for line, (_, line_h, baseline) in zip(lines, metrics):
        cursor_y += line_h
        cv2.putText(
            frame_bgr,
            line,
            (label_x + padding_x, cursor_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += baseline + line_gap
