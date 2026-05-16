"""Age and gender prediction using Keras models.

Backbone: EfficientNetB2 (has built-in preprocessing).
Input images are normalised to [0, 1] before prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tensorflow import keras

from .utils import preprocess_face, softmax


logger = logging.getLogger(__name__)

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_LABELS = ["Female", "Male"]


def _resolve_model_path(models_dir: Path, filename: str, fallback_subdir: str) -> Path:
    """Find a model file in the expected subdirectory or models/ root."""
    direct_path = models_dir / filename
    nested_path = models_dir / fallback_subdir / filename
    if nested_path.exists():
        return nested_path
    if direct_path.exists():
        return direct_path
    return direct_path


def load_age_gender_models(models_dir: str | Path) -> Tuple[keras.Model, keras.Model]:
    """Load gender and age Keras models from *models_dir*.

    Returns
    -------
    tuple[keras.Model, keras.Model]
        ``(gender_model, age_model)``

    Raises
    ------
    FileNotFoundError
        If one or both ``.keras`` files are missing.
    """
    models_dir = Path(models_dir)
    gender_path = _resolve_model_path(models_dir, "gender_model.keras", "age_gender")
    age_path = _resolve_model_path(models_dir, "age_model.keras", "age_gender")

    missing = [str(path) for path in [gender_path, age_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing age/gender model file(s): "
            + ", ".join(missing)
            + ". Train notebooks first, then copy .keras files into models/."
        )

    logger.info("Loading gender model: %s", gender_path)
    gender_model = keras.models.load_model(gender_path, compile=False)
    logger.info("Loading age model: %s", age_path)
    age_model = keras.models.load_model(age_path, compile=False)
    return gender_model, age_model


def _to_probability_vector(output: np.ndarray, class_count: int) -> np.ndarray:
    """Convert raw model output to a valid probability vector.

    Handles three cases:
    1. Binary classification with a single sigmoid output (class_count == 2).
    2. Output already looks like softmax probabilities.
    3. Raw logits that need softmax.
    """
    output = np.asarray(output)
    if output.ndim > 1:
        output = output[0]

    # Binary head with single sigmoid neuron
    if class_count == 2 and output.size == 1:
        male_prob = float(output.reshape(-1)[0])
        male_prob = max(0.0, min(1.0, male_prob))
        return np.array([1.0 - male_prob, male_prob], dtype="float32")

    if output.size != class_count:
        raise ValueError(f"Expected {class_count} outputs, got shape {output.shape}")

    # Already valid probabilities
    if np.min(output) >= 0.0 and np.max(output) <= 1.0 and np.isclose(np.sum(output), 1.0, atol=1e-3):
        return output.astype("float32")

    return softmax(output)


def predict_age_gender(
    face_bgr: np.ndarray,
    gender_model: keras.Model,
    age_model: keras.Model,
    image_size: int = 224,
) -> Dict[str, object]:
    """Predict age group and gender for a single BGR face crop.

    The face is preprocessed to ``[0, 1]`` range which is compatible
    with EfficientNetB2's built-in preprocessing.

    Returns
    -------
    dict
        Keys: ``gender``, ``gender_confidence``, ``age``, ``age_confidence``.
    """
    batch = preprocess_face(face_bgr, image_size)

    gender_output = gender_model.predict(batch, verbose=0)
    age_output = age_model.predict(batch, verbose=0)

    gender_probs = _to_probability_vector(gender_output, len(GENDER_LABELS))
    age_probs = _to_probability_vector(age_output, len(AGE_LABELS))

    gender_idx = int(np.argmax(gender_probs))
    age_idx = int(np.argmax(age_probs))

    return {
        "gender": GENDER_LABELS[gender_idx],
        "gender_confidence": float(gender_probs[gender_idx]),
        "age": AGE_LABELS[age_idx],
        "age_confidence": float(age_probs[age_idx]),
    }
