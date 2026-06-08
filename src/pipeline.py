"""Shared inference pipeline for desktop and web entrypoints."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .detect_face import Box, FaceDetectionConfig, crop_face_bgr, detect_faces_bgr, draw_prediction
from .predict_age_gender import load_age_gender_models, predict_age_gender
from .predict_emotion import load_emotion_model, predict_emotion


logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
IMAGE_SIZE = 224


@dataclass
class Models:
    """Container for loaded Keras models."""

    gender_model: object
    age_model: object
    emotion_model: object


@dataclass
class FaceResult:
    """Prediction result for a single face."""

    gender: str
    gender_confidence: float
    age: str
    age_confidence: float
    emotion: str
    emotion_confidence: float


@dataclass
class FacePrediction:
    """A detected face box with its predicted attributes."""

    box: Box
    result: FaceResult


def load_models(models_dir: str | Path = MODELS_DIR) -> Models:
    """Load all Keras models from the models directory."""
    gender_model, age_model = load_age_gender_models(models_dir)
    emotion_model = load_emotion_model(models_dir)
    models = Models(
        gender_model=gender_model,
        age_model=age_model,
        emotion_model=emotion_model,
    )
    warm_up_models(models)
    return models


def warm_up_models(models: Models) -> None:
    """Run one dummy inference so first user-facing prediction is faster."""
    dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype="float32")
    try:
        models.gender_model.predict(dummy, verbose=0)
        models.age_model.predict(dummy, verbose=0)
        models.emotion_model.predict(dummy, verbose=0)
    except Exception as exc:
        logger.warning("Could not warm up models: %s", exc)


def predict_face(face_bgr: np.ndarray, models: Models) -> FaceResult:
    """Run age, gender, and emotion prediction on one face crop."""
    age_gender = predict_age_gender(
        face_bgr,
        models.gender_model,
        models.age_model,
        image_size=IMAGE_SIZE,
    )
    emotion = predict_emotion(
        face_bgr,
        models.emotion_model,
        image_size=IMAGE_SIZE,
    )
    return FaceResult(
        gender=age_gender["gender"],
        gender_confidence=age_gender["gender_confidence"],
        age=age_gender["age"],
        age_confidence=age_gender["age_confidence"],
        emotion=emotion["emotion"],
        emotion_confidence=emotion["emotion_confidence"],
    )


def analyze_frame_bgr(
    frame_bgr: np.ndarray,
    models: Models,
    config: FaceDetectionConfig,
    *,
    max_faces: int | None = None,
    annotate: bool = True,
) -> List[FacePrediction]:
    """Detect faces, run predictions, and draw annotations in-place."""
    boxes = detect_faces_bgr(frame_bgr, config)
    if max_faces is not None:
        boxes = boxes[: max(0, int(max_faces))]

    predictions: List[FacePrediction] = []
    for box in boxes:
        try:
            face = crop_face_bgr(frame_bgr, box, margin=config.margin)
            result = predict_face(face, models)
        except Exception as exc:
            logger.warning("Skipping face with prediction error: %s", exc)
            continue

        if annotate:
            draw_prediction(
                frame_bgr,
                box,
                age_label=result.age,
                gender_label=result.gender,
                emotion_label=result.emotion,
                score=result.emotion_confidence,
            )
        predictions.append(FacePrediction(box=box, result=result))
    return predictions
