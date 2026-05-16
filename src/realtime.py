"""Helpers for lower-latency realtime face inference."""
from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable, List

import cv2
import numpy as np

from .detect_face import Box, FaceDetectionConfig, crop_face_bgr, detect_faces_bgr, draw_prediction


logger = logging.getLogger(__name__)


@dataclass
class CachedFacePrediction:
    """Face box and prediction result cached from the last inference frame."""

    box: Box
    result: object


class AsyncRealtimeFaceProcessor:
    """Non-blocking realtime processor for smoother camera display.

    ``process_frame`` only submits the newest frame for background inference
    and draws the latest cached result. The camera loop no longer pauses while
    TensorFlow is predicting.
    """

    def __init__(
        self,
        models: object,
        predict_face: Callable[[np.ndarray, object], object],
        config: FaceDetectionConfig | None = None,
        *,
        predict_emotion: Callable[[np.ndarray, object], tuple[str, float]] | None = None,
        partial_result_factory: Callable[[str, float], object] | None = None,
        inference_interval: float = 0.25,
        full_inference_interval: float = 1.20,
        detection_width: int = 360,
        max_faces: int = 1,
        emotion_history_size: int = 5,
        emotion_min_confidence: float = 0.32,
        label_switch_margin: float = 0.18,
        label_switch_count: int = 3,
    ) -> None:
        self.models = models
        self.predict_face = predict_face
        self.config = config or FaceDetectionConfig()
        self.predict_emotion = predict_emotion
        self.partial_result_factory = partial_result_factory
        self.inference_interval = max(0.05, float(inference_interval))
        self.full_inference_interval = max(0.3, float(full_inference_interval))
        self.detection_width = max(160, int(detection_width))
        self.max_faces = max(1, int(max_faces))
        self.emotion_history_size = max(1, int(emotion_history_size))
        self.emotion_min_confidence = float(emotion_min_confidence)
        self.label_switch_margin = float(label_switch_margin)
        self.label_switch_count = max(1, int(label_switch_count))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._last_submit_time = 0.0
        self._last_full_predict_time = time.time()
        self._worker_thread: threading.Thread | None = None
        self.cached_predictions: List[CachedFacePrediction] = []
        self._emotion_histories: list[deque[tuple[str, float]]] = [
            deque(maxlen=self.emotion_history_size) for _ in range(self.max_faces)
        ]
        self._stable_emotions: list[tuple[str, float] | None] = [
            None for _ in range(self.max_faces)
        ]

    def start(self) -> None:
        """Start the background inference worker."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background inference worker."""
        self._stop_event.set()
        self._wake_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)

    def process_frame(self, frame_bgr: np.ndarray, *, force: bool = False) -> int:
        """Submit frame for inference and draw cached labels immediately."""
        self._submit_frame(frame_bgr, force=force)
        predictions = self._get_cached_predictions()

        for cached in predictions:
            result = cached.result
            draw_prediction(
                frame_bgr,
                cached.box,
                age_label=result.age,
                gender_label=result.gender,
                emotion_label=result.emotion,
                score=result.emotion_confidence,
            )

        return len(predictions)

    def _submit_frame(self, frame_bgr: np.ndarray, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_submit_time < self.inference_interval:
            return
        self._last_submit_time = now

        with self._lock:
            self._latest_frame = frame_bgr.copy()
        self._wake_event.set()

    def _get_cached_predictions(self) -> List[CachedFacePrediction]:
        with self._lock:
            return list(self.cached_predictions)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.1)
            self._wake_event.clear()

            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            predictions = self._predict_current_frame(frame)
            predictions = self._smooth_predictions(predictions)

            with self._lock:
                self.cached_predictions = predictions

    def _predict_current_frame(self, frame_bgr: np.ndarray) -> List[CachedFacePrediction]:
        boxes = self._detect_boxes(frame_bgr)
        predictions: List[CachedFacePrediction] = []
        now = time.time()
        run_full = (
            self.predict_emotion is None
            or now - self._last_full_predict_time >= self.full_inference_interval
        )

        with self._lock:
            previous_predictions = list(self.cached_predictions)

        for index, box in enumerate(boxes[: self.max_faces]):
            try:
                face = crop_face_bgr(frame_bgr, box, margin=self.config.margin)
                if run_full:
                    result = self.predict_face(face, self.models)
                else:
                    result = self._predict_fast_result(face, previous_predictions, index)
            except Exception as exc:
                logger.warning("Skipping face with prediction error: %s", exc)
                continue
            predictions.append(CachedFacePrediction(box=box, result=result))

        if run_full and predictions:
            self._last_full_predict_time = now

        return predictions

    def _detect_boxes(self, frame_bgr: np.ndarray) -> List[Box]:
        height, width = frame_bgr.shape[:2]
        if width <= self.detection_width:
            return detect_faces_bgr(frame_bgr, self.config)

        scale = self.detection_width / float(width)
        resized = cv2.resize(
            frame_bgr,
            (self.detection_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        resized_boxes = detect_faces_bgr(resized, self.config)
        inv_scale = 1.0 / scale

        boxes: List[Box] = []
        for x, y, w, h in resized_boxes:
            boxes.append(
                (
                    int(x * inv_scale),
                    int(y * inv_scale),
                    int(w * inv_scale),
                    int(h * inv_scale),
                )
            )
        return boxes

    def _predict_fast_result(
        self,
        face_bgr: np.ndarray,
        previous_predictions: List[CachedFacePrediction],
        index: int,
    ) -> object:
        if self.predict_emotion is None:
            return self.predict_face(face_bgr, self.models)

        emotion, confidence = self.predict_emotion(face_bgr, self.models)

        if index < len(previous_predictions):
            result = copy(previous_predictions[index].result)
        elif self.partial_result_factory is not None:
            result = self.partial_result_factory(emotion, confidence)
        else:
            result = self.predict_face(face_bgr, self.models)

        result.emotion = emotion
        result.emotion_confidence = confidence
        return result

    def _smooth_predictions(
        self,
        predictions: List[CachedFacePrediction],
    ) -> List[CachedFacePrediction]:
        if not predictions:
            for history in self._emotion_histories:
                history.clear()
            self._stable_emotions = [None for _ in range(self.max_faces)]
            return predictions

        for index, cached in enumerate(predictions[: self.max_faces]):
            result = cached.result
            label = str(result.emotion)
            confidence = float(result.emotion_confidence)
            history = self._emotion_histories[index]

            if confidence < self.emotion_min_confidence and self._stable_emotions[index]:
                label, confidence = self._stable_emotions[index]
            else:
                history.append((label, confidence))
                label, confidence = self._stable_emotion(index)

            result.emotion = label
            result.emotion_confidence = confidence
        return predictions

    def _stable_emotion(self, index: int) -> tuple[str, float]:
        history = self._emotion_histories[index]
        candidate_label, candidate_confidence = self._majority_emotion(history)
        stable = self._stable_emotions[index]
        if stable is None:
            self._stable_emotions[index] = (candidate_label, candidate_confidence)
            return candidate_label, candidate_confidence

        stable_label, stable_confidence = stable
        recent_candidate_count = sum(
            1 for label, _confidence in history if label == candidate_label
        )
        should_switch = (
            candidate_label == stable_label
            or recent_candidate_count >= self.label_switch_count
            or candidate_confidence >= stable_confidence + self.label_switch_margin
        )

        if should_switch:
            self._stable_emotions[index] = (candidate_label, candidate_confidence)
            return candidate_label, candidate_confidence
        return stable_label, stable_confidence

    def _majority_emotion(self, history: deque[tuple[str, float]]) -> tuple[str, float]:
        scores: dict[str, float] = {}
        for history_index, (label, confidence) in enumerate(history, start=1):
            scores[label] = scores.get(label, 0.0) + confidence * history_index

        label = max(scores, key=scores.get)
        confidences = [
            confidence
            for history_label, confidence in history
            if history_label == label
        ]
        return label, float(sum(confidences) / max(1, len(confidences)))
