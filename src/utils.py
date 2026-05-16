"""Shared utility functions for face prediction modules.

This module eliminates code duplication between predict_age_gender.py
and predict_emotion.py by centralizing common preprocessing logic.
"""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_face(face_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """Convert a BGR face crop to a normalised RGB batch tensor.

    The output is in the range [0, 1] which is compatible with
    EfficientNetB2's built-in preprocessing layer.

    Parameters
    ----------
    face_bgr:
        Face crop as an OpenCV BGR ``np.ndarray``.  Must be non-empty.
    image_size:
        Target spatial size (both width and height).

    Returns
    -------
    np.ndarray
        Float32 tensor with shape ``(1, image_size, image_size, 3)``.

    Raises
    ------
    ValueError
        If *face_bgr* has zero pixels.
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


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis.

    Parameters
    ----------
    values:
        Raw logits (any shape).

    Returns
    -------
    np.ndarray
        Probabilities with the same shape, summing to 1 along ``axis=-1``.
    """
    values = values.astype("float32")
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)
