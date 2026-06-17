from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass
class AnalyticsFrameSnapshot:
    timestamp: float
    face_count: int
    emotion_counts: dict[str, int]
    gender_counts: dict[str, int]
    age_counts: dict[str, int]


class RealtimeAnalyticsStore:
    """Keeps lightweight rolling analytics for webcam predictions."""

    def __init__(self, *, window_seconds: int = 60, sample_interval: float = 1.0) -> None:
        self.window_seconds = max(10, int(window_seconds))
        self.sample_interval = max(0.25, float(sample_interval))
        self.history: deque[AnalyticsFrameSnapshot] = deque()
        self.emotion_totals: Counter[str] = Counter()
        self.current = AnalyticsFrameSnapshot(
            timestamp=0.0,
            face_count=0,
            emotion_counts={},
            gender_counts={},
            age_counts={},
        )
        self._last_sample_time = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.emotion_totals.clear()
        self.current = AnalyticsFrameSnapshot(
            timestamp=0.0,
            face_count=0,
            emotion_counts={},
            gender_counts={},
            age_counts={},
        )
        self._last_sample_time = 0.0

    def update(self, predictions: Iterable[object], *, timestamp: float) -> None:
        emotion_counts: Counter[str] = Counter()
        gender_counts: Counter[str] = Counter()
        age_counts: Counter[str] = Counter()
        face_count = 0

        for prediction in predictions:
            result = getattr(prediction, 'result', None)
            if result is None:
                continue
            face_count += 1

            emotion = str(getattr(result, 'emotion', '') or '').strip()
            if emotion and emotion != '...':
                emotion_counts[emotion] += 1
                self.emotion_totals[emotion] += 1

            gender = str(getattr(result, 'gender', '') or '').strip()
            if gender and gender != '...':
                gender_counts[gender] += 1

            age = str(getattr(result, 'age', '') or '').strip()
            if age and age != '...':
                age_counts[age] += 1

        self.current = AnalyticsFrameSnapshot(
            timestamp=timestamp,
            face_count=face_count,
            emotion_counts=dict(emotion_counts),
            gender_counts=dict(gender_counts),
            age_counts=dict(age_counts),
        )

        if timestamp - self._last_sample_time < self.sample_interval:
            self._trim_history(timestamp)
            return

        self.history.append(
            AnalyticsFrameSnapshot(
                timestamp=timestamp,
                face_count=face_count,
                emotion_counts=dict(emotion_counts),
                gender_counts=dict(gender_counts),
                age_counts=dict(age_counts),
            )
        )
        self._last_sample_time = timestamp
        self._trim_history(timestamp)

    def _trim_history(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

    def get_current_summary(self) -> AnalyticsFrameSnapshot:
        return self.current

    def get_gender_ratio_text(self) -> str:
        counts = self.current.gender_counts
        total = sum(counts.values())
        if total <= 0:
            return '--'
        parts = []
        for label in ('Male', 'Female'):
            value = counts.get(label, 0)
            if value:
                pct = (value / total) * 100.0
                parts.append(f'{label} {pct:.0f}%')
        if not parts:
            for label, value in counts.items():
                pct = (value / total) * 100.0
                parts.append(f'{label} {pct:.0f}%')
        return ' / '.join(parts)

    def get_age_summary_text(self, *, limit: int = 3) -> str:
        counts = self.current.age_counts
        if not counts:
            return '--'
        items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(1, int(limit))]
        return ', '.join(f'{label}: {value}' for label, value in items)

    def get_emotion_totals(self) -> dict[str, int]:
        return dict(self.emotion_totals)

    def get_emotion_trend_series(self) -> tuple[list[str], dict[str, list[int]]]:
        snapshots = list(self.history)
        labels = [self._format_time_offset(item.timestamp, snapshots[-1].timestamp) for item in snapshots] if snapshots else []
        emotion_keys: list[str] = []
        seen: set[str] = set()
        for item in snapshots:
            for key in item.emotion_counts:
                if key not in seen:
                    seen.add(key)
                    emotion_keys.append(key)
        series = {key: [item.emotion_counts.get(key, 0) for item in snapshots] for key in emotion_keys}
        return labels, series

    @staticmethod
    def _format_time_offset(ts: float, end_ts: float) -> str:
        delta = max(0, int(round(end_ts - ts)))
        return f'-{delta}s'
