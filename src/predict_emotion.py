"""Emotion prediction using a Keras model (7 classes).

Classes: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger.
Backbone: EfficientNetB2 (has built-in preprocessing).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from tensorflow import keras

from .utils import softmax


logger = logging.getLogger(__name__)

EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
HAPPY_INDEX = EMOTION_LABELS.index("Happy")


def _load_smile_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        logger.warning("Cannot load OpenCV smile cascade: %s", cascade_path)
    return cascade


SMILE_CASCADE = _load_smile_cascade()


def load_emotion_model(models_dir: str | Path) -> keras.Model:
    """Load the emotion Keras model from *models_dir*.

    Raises
    ------
    FileNotFoundError
        If ``emotion_model.keras`` is missing.
    """
    models_dir = Path(models_dir)
    model_path = models_dir / "emotion_model.keras"
    nested_path = models_dir / "emotion" / "emotion_model.keras"
    if nested_path.exists():
        model_path = nested_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing emotion model file: {model_path}. "
            "Train notebooks first, then copy emotion_model.keras into models/."
        )
    logger.info("Loading emotion model: %s", model_path)
    return keras.models.load_model(model_path, compile=False)


def _preprocess_face(face_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """Preprocess face for EfficientNetB2.

    EfficientNetB2 has built-in preprocessing that expects pixel values
    in [0, 255] range. We send uint8 values directly.
    """
    if face_bgr.size == 0:
        raise ValueError("Cannot preprocess an empty face image.")

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(
        face_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    face_rgb = face_rgb.astype("float32") / 255.0
    return np.expand_dims(face_rgb, axis=0)


def _to_probability_vector(output: np.ndarray) -> np.ndarray:
    """Convert raw model output to a valid probability vector."""
    output = np.asarray(output)
    if output.ndim > 1:
        output = output[0]
    if output.size != len(EMOTION_LABELS):
        raise ValueError(f"Expected {len(EMOTION_LABELS)} emotion outputs, got shape {output.shape}")
    # Already valid probabilities
    if np.min(output) >= 0.0 and np.max(output) <= 1.0 and np.isclose(np.sum(output), 1.0, atol=1e-3):
        return output.astype("float32")
    return softmax(output)


def detect_smile_bgr(face_bgr: np.ndarray) -> bool:
    """Detect a clear smile in a face crop using OpenCV Haar cascade."""
    if face_bgr.size == 0 or SMILE_CASCADE.empty():
        return False

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    height, width = gray.shape[:2]

    lower_face_y = int(height * 0.38)
    lower_face = gray[lower_face_y:, :]
    min_size = (max(24, int(width * 0.22)), max(10, int(height * 0.08)))
    smiles = SMILE_CASCADE.detectMultiScale(
        lower_face,
        scaleFactor=1.6,
        minNeighbors=18,
        minSize=min_size,
    )

    for _, _, smile_w, smile_h in smiles:
        if smile_w >= width * 0.24 and smile_h >= height * 0.07:
            return True
    return False


def predict_emotion(
    face_bgr: np.ndarray,
    emotion_model: keras.Model,
    image_size: int = 224,
) -> Dict[str, object]:
    """Predict emotion for a single BGR face crop.

    Returns
    -------
    dict
        Keys: ``emotion``, ``emotion_confidence``.
    """
    batch = _preprocess_face(face_bgr, image_size)
    output = emotion_model.predict(batch, verbose=0)
    probs = _to_probability_vector(output)
    idx = int(np.argmax(probs))
    smile_detected = detect_smile_bgr(face_bgr)

    if smile_detected and idx != HAPPY_INDEX:
        top_confidence = float(probs[idx])
        happy_confidence = float(probs[HAPPY_INDEX])
        if happy_confidence >= 0.08 or top_confidence < 0.95:
            idx = HAPPY_INDEX
            probs = probs.copy()
            probs[HAPPY_INDEX] = max(happy_confidence, 0.72)

    return {
        "emotion": EMOTION_LABELS[idx],
        "emotion_confidence": float(probs[idx]),
        "smile_detected": smile_detected,
    }
