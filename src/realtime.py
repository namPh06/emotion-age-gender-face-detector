"""Helpers for lower-latency realtime face inference.

Architecture: Two-phase background processing.
- Phase 1 (every frame): Fast face detection (~15ms at 320px) updates
  bounding box positions immediately so boxes track faces smoothly.
- Phase 2 (throttled): ML prediction (age/gender/emotion) updates labels
  at a lower rate without blocking the detection cadence.

This decouples box tracking from prediction latency, eliminating the
visual lag where boxes would freeze while models were running.
"""
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

    Uses a two-phase background loop:
    - **Detection phase** runs on every submitted frame for responsive
      bounding box tracking (~15ms at 320px).
    - **Prediction phase** runs at a lower rate to update age, gender,
      and emotion labels without blocking the detection cadence.

    The camera loop calls ``process_frame`` which submits the frame and
    draws the latest merged results (fresh boxes + cached labels).
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
        self._last_full_predict_time = time.time()
        self._last_predict_time = 0.0
        self._worker_thread: threading.Thread | None = None
        # cached_predictions: latest ML labels (used for label matching)
        self.cached_predictions: List[CachedFacePrediction] = []
        # _display_predictions: fresh boxes merged with cached labels (used for drawing)
        self._display_predictions: List[CachedFacePrediction] = []
        self._prev_boxes: List[Box] = []
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
        """Submit frame for background processing.

        No throttling — always accepts the latest frame so detection
        runs as fast as the worker can process.  The worker naturally
        drops stale frames by always picking the newest one.
        """
        with self._lock:
            self._latest_frame = frame_bgr.copy()
        self._wake_event.set()

    def _get_cached_predictions(self) -> List[CachedFacePrediction]:
        """Return display predictions (fresh boxes + cached labels)."""
        with self._lock:
            return list(self._display_predictions)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Two-phase loop: always detect, conditionally predict."""
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.04)
            self._wake_event.clear()

            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            # Phase 1: ALWAYS detect faces (fast at reduced resolution)
            boxes = self._detect_boxes(frame)
            boxes = boxes[: self.max_faces]

            # Immediately merge fresh boxes with cached labels for display
            with self._lock:
                cached_labels = list(self.cached_predictions)
            display = self._merge_boxes_with_labels(boxes, cached_labels)
            with self._lock:
                self._display_predictions = display

            # Phase 2: Conditionally run ML predictions (throttled)
            now = time.time()
            run_full = (
                self.predict_emotion is None
                or now - self._last_full_predict_time >= self.full_inference_interval
            )
            should_predict = run_full or (
                now - self._last_predict_time >= self.inference_interval
            )

            if should_predict and boxes:
                predictions = self._predict_on_boxes(
                    frame, boxes, cached_labels, run_full
                )
                predictions = self._smooth_predictions(predictions)

                with self._lock:
                    self.cached_predictions = predictions
                    self._display_predictions = predictions

                self._last_predict_time = now
                if run_full and predictions:
                    self._last_full_predict_time = now

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

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
                    round(x * inv_scale),
                    round(y * inv_scale),
                    round(w * inv_scale),
                    round(h * inv_scale),
                )
            )
        return boxes

    def _merge_boxes_with_labels(
        self,
        fresh_boxes: List[Box],
        cached_predictions: List[CachedFacePrediction],
    ) -> List[CachedFacePrediction]:
        """Match fresh detection boxes with cached prediction labels via IoU.

        This is the key to smooth box tracking: detection boxes are always
        fresh (from the current frame) while labels come from the latest
        prediction cycle.
        """
        if not fresh_boxes:
            return []

        if not cached_predictions:
            # No labels yet — show boxes with placeholder labels if possible
            result: List[CachedFacePrediction] = []
            for box in fresh_boxes:
                if self.partial_result_factory:
                    placeholder = self.partial_result_factory("...", 0.0)
                    result.append(CachedFacePrediction(box=box, result=placeholder))
            return result

        result: List[CachedFacePrediction] = []
        used: set[int] = set()
        for box in fresh_boxes:
            best_idx: int | None = None
            best_iou = 0.08  # Low threshold to match even when face moves fast
            for i, cached in enumerate(cached_predictions):
                if i in used:
                    continue
                iou = self._box_iou(box, cached.box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx is not None:
                used.add(best_idx)
                result.append(
                    CachedFacePrediction(
                        box=box,
                        result=copy(cached_predictions[best_idx].result),
                    )
                )
            elif self.partial_result_factory:
                placeholder = self.partial_result_factory("...", 0.0)
                result.append(CachedFacePrediction(box=box, result=placeholder))

        return result

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_on_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes: List[Box],
        previous_predictions: List[CachedFacePrediction],
        run_full: bool,
    ) -> List[CachedFacePrediction]:
        """Run ML predictions on detected face boxes."""
        predictions: List[CachedFacePrediction] = []
        for box in boxes:
            try:
                face = crop_face_bgr(frame_bgr, box, margin=self.config.margin)
                if run_full:
                    result = self.predict_face(face, self.models)
                else:
                    result = self._predict_fast_result(
                        face, box, previous_predictions
                    )
            except Exception as exc:
                logger.warning("Skipping face with prediction error: %s", exc)
                continue
            predictions.append(CachedFacePrediction(box=box, result=result))
        return predictions

    def _predict_fast_result(
        self,
        face_bgr: np.ndarray,
        current_box: Box,
        previous_predictions: List[CachedFacePrediction],
    ) -> object:
        if self.predict_emotion is None:
            return self.predict_face(face_bgr, self.models)

        emotion, confidence = self.predict_emotion(face_bgr, self.models)

        # Match by IoU instead of index to handle face reordering
        best_prev = self._find_best_iou_match(current_box, previous_predictions)
        if best_prev is not None:
            result = copy(best_prev.result)
        elif self.partial_result_factory is not None:
            result = self.partial_result_factory(emotion, confidence)
        else:
            result = self.predict_face(face_bgr, self.models)

        result.emotion = emotion
        result.emotion_confidence = confidence
        return result

    # ------------------------------------------------------------------
    # IoU / matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _box_iou(a: Box, b: Box) -> float:
        """Compute Intersection-over-Union between two (x, y, w, h) boxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1e-6)

    def _find_best_iou_match(
        self,
        box: Box,
        previous_predictions: List[CachedFacePrediction],
        min_iou: float = 0.25,
    ) -> CachedFacePrediction | None:
        """Find the previous prediction whose box overlaps most with *box*."""
        best: CachedFacePrediction | None = None
        best_iou = min_iou
        for prev in previous_predictions:
            iou = self._box_iou(box, prev.box)
            if iou > best_iou:
                best_iou = iou
                best = prev
        return best

    # ------------------------------------------------------------------
    # Emotion smoothing
    # ------------------------------------------------------------------

    def _smooth_predictions(
        self,
        predictions: List[CachedFacePrediction],
    ) -> List[CachedFacePrediction]:
        if not predictions:
            for history in self._emotion_histories:
                history.clear()
            self._stable_emotions = [None for _ in range(self.max_faces)]
            self._prev_boxes = []
            return predictions

        # Match current predictions to history slots by IoU with previous boxes
        slot_map = self._match_to_history_slots(predictions)

        for pred_idx, cached in enumerate(predictions[: self.max_faces]):
            hist_idx = slot_map[pred_idx]
            result = cached.result
            label = str(result.emotion)
            confidence = float(result.emotion_confidence)
            history = self._emotion_histories[hist_idx]

            if confidence < self.emotion_min_confidence and self._stable_emotions[hist_idx]:
                label, confidence = self._stable_emotions[hist_idx]
            else:
                history.append((label, confidence))
                label, confidence = self._stable_emotion(hist_idx)

            result.emotion = label
            result.emotion_confidence = confidence

        # Clear unused history slots
        used_slots = set(slot_map.values())
        for i in range(self.max_faces):
            if i not in used_slots:
                self._emotion_histories[i].clear()
                self._stable_emotions[i] = None

        # Update tracked boxes for next frame
        self._prev_boxes = [cached.box for cached in predictions[: self.max_faces]]
        return predictions

    def _match_to_history_slots(
        self,
        predictions: List[CachedFacePrediction],
    ) -> dict[int, int]:
        """Map each prediction index to a history slot using IoU matching."""
        slot_map: dict[int, int] = {}
        used_slots: set[int] = set()

        for pred_idx, cached in enumerate(predictions[: self.max_faces]):
            best_slot: int | None = None
            best_iou = 0.20

            for prev_idx, prev_box in enumerate(self._prev_boxes):
                if prev_idx in used_slots or prev_idx >= self.max_faces:
                    continue
                iou = self._box_iou(cached.box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_slot = prev_idx

            if best_slot is not None:
                slot_map[pred_idx] = best_slot
                used_slots.add(best_slot)
            else:
                # Assign to first free slot and reset its history
                for slot in range(self.max_faces):
                    if slot not in used_slots:
                        slot_map[pred_idx] = slot
                        used_slots.add(slot)
                        self._emotion_histories[slot].clear()
                        self._stable_emotions[slot] = None
                        break

        return slot_map

    def _stable_emotion(self, index: int) -> tuple[str, float]:
        history = self._emotion_histories[index]
        candidate_label, candidate_confidence = self._majority_emotion(history)
        stable = self._stable_emotions[index]
        if stable is None:
            self._stable_emotions[index] = (candidate_label, candidate_confidence)
            return candidate_label, candidate_confidence

        stable_label, stable_confidence = stable

        # Fast path: same emotion — always update confidence
        if candidate_label == stable_label:
            self._stable_emotions[index] = (candidate_label, candidate_confidence)
            return candidate_label, candidate_confidence

        # Different emotion — check switch conditions
        recent_candidate_count = sum(
            1 for label, _confidence in history if label == candidate_label
        )
        should_switch = (
            recent_candidate_count >= self.label_switch_count
            or candidate_confidence >= stable_confidence + self.label_switch_margin
        )

        if should_switch:
            self._stable_emotions[index] = (candidate_label, candidate_confidence)
            return candidate_label, candidate_confidence
        return stable_label, stable_confidence

    def _majority_emotion(self, history: deque[tuple[str, float]]) -> tuple[str, float]:
        """Weighted majority vote with exponential recency bias.

        Uses ``2 ** index`` weighting so the newest reading dominates,
        making emotion transitions more responsive while still filtering
        single-frame noise via the history window.
        """
        scores: dict[str, float] = {}
        for history_index, (label, confidence) in enumerate(history):
            weight = 2 ** history_index  # exponential: 1, 2, 4, 8, ...
            scores[label] = scores.get(label, 0.0) + confidence * weight

        label = max(scores, key=scores.get)
        confidences = [
            confidence
            for history_label, confidence in history
            if history_label == label
        ]
        return label, float(sum(confidences) / max(1, len(confidences)))
