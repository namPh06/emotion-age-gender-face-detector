"""Face detection, cropping, and annotation utilities.

Uses OpenCV's Haar Cascade classifier for face detection.
Future upgrade: replace with ``cv2.dnn`` (ResNet-SSD) or ``mediapipe``
for better accuracy on rotated / poorly-lit faces.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


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
    """
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (48, 48)
    margin: float = 0.35


def _load_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Cannot load OpenCV Haar cascade: {cascade_path}")
    return cascade


FACE_CASCADE = _load_cascade()


def detect_faces_bgr(
    frame_bgr: np.ndarray,
    config: FaceDetectionConfig | None = None,
) -> List[Box]:
    """Detect faces in a BGR image and return boxes as ``(x, y, w, h)``.

    Boxes are sorted by area in descending order (largest face first).
    """
    if config is None:
        config = FaceDetectionConfig()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=config.scale_factor,
        minNeighbors=config.min_neighbors,
        minSize=config.min_size,
    )
    boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)


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
    label = f"{gender_label} | {age_label} | {emotion_label}"
    if score is not None:
        label = f"{label} | {score:.2f}"

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 220, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    label_y = max(0, y - text_h - baseline - 8)

    cv2.rectangle(
        frame_bgr,
        (x, label_y),
        (min(frame_bgr.shape[1] - 1, x + text_w + 10), label_y + text_h + baseline + 8),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame_bgr,
        label,
        (x + 5, label_y + text_h + 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
