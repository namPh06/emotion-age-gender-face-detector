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
    score_text = f" | {score:.2f}" if score is not None else ""

    # Prefer a single line when it fits, but fall back to shorter multi-line
    # layouts for small images so the annotation is not clipped.
    label_variants: list[list[str]] = [
        [f"{gender_label} | {age_label} | {emotion_label}{score_text}"],
        [f"{gender_label} | {age_label}", f"{emotion_label}{score_text}"],
        [gender_label, age_label, f"{emotion_label}{score_text}"],
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
