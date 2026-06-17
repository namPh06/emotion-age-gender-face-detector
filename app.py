"""Desktop UI for face age, gender, and emotion detection."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from src.detect_face import FaceDetectionConfig, crop_face_bgr, detect_faces_bgr, draw_prediction
from src.predict_age_gender import load_age_gender_models, predict_age_gender
from src.predict_emotion import load_emotion_model, predict_emotion
from src.realtime import AsyncRealtimeFaceProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DISPLAY_SIZE = (960, 640)
CAMERA_SIZE = (640, 480)
DISPLAY_INTERVAL = 1.0 / 30.0
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
IMAGE_SIZE = 224
HISTORY_LIMIT = 10


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
    """A detected face with prediction details for the desktop UI."""

    box: tuple[int, int, int, int]
    result: FaceResult
    crop_bgr: np.ndarray | None = None


@dataclass
class AnalysisResult:
    """A still-image or snapshot result stored for display/history."""

    source: str
    frame_bgr: np.ndarray
    frame_width: int
    frame_height: int
    face_count: int
    elapsed_ms: int
    detector: str
    quality: str
    predictions: list[FacePrediction]
    saved_path: Path | None = None
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class ScrollableFrame(ttk.Frame):
    """A lightweight scrollable frame for result cards."""

    def __init__(self, parent, *, height: int = 260) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self.canvas = tk.Canvas(
            self,
            height=height,
            bg="#161f30",
            bd=0,
            highlightthickness=0,
        )
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Panel.TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor=tk.NW)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)

        # Bind mousewheel scrolling
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _update_scroll_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def load_models() -> Models:
    """Load all Keras models from the models directory."""
    gender_model, age_model = load_age_gender_models(MODELS_DIR)
    emotion_model = load_emotion_model(MODELS_DIR)
    models = Models(
        gender_model=gender_model,
        age_model=age_model,
        emotion_model=emotion_model,
    )
    warm_up_models(models)
    return models


def warm_up_models(models: Models) -> None:
    """Run one dummy inference so webcam labels appear faster later."""
    dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype="float32")
    try:
        models.gender_model(dummy, training=False)
        models.age_model(dummy, training=False)
        models.emotion_model(dummy, training=False)
    except Exception as exc:
        logger.warning("Could not warm up models: %s", exc)


def predict_face(face_bgr: np.ndarray, models: Models) -> FaceResult:
    """Run all predictions on one face crop.

    Preprocesses the face once and reuses the batch for all three models
    to avoid redundant BGR→RGB conversion, resize, and float normalisation.
    """
    from src.utils import preprocess_face as _preprocess
    from src.predict_age_gender import _to_probability_vector as _ag_probs, AGE_LABELS, GENDER_LABELS
    from src.predict_emotion import (
        _preprocess_face as _preprocess_emotion,
        _to_probability_vector as _em_probs,
        EMOTION_LABELS, HAPPY_INDEX, detect_smile_bgr,
    )

    # --- Age / Gender (shared preprocessing) ---
    batch = _preprocess(face_bgr, IMAGE_SIZE)
    gender_output = models.gender_model(batch, training=False)
    age_output = models.age_model(batch, training=False)
    gender_probs = _ag_probs(gender_output, len(GENDER_LABELS))
    age_probs = _ag_probs(age_output, len(AGE_LABELS))
    gender_idx = int(np.argmax(gender_probs))
    age_idx = int(np.argmax(age_probs))

    # --- Emotion (uses its own preprocessing) ---
    emotion_batch = _preprocess_emotion(face_bgr, IMAGE_SIZE)
    emotion_output = models.emotion_model(emotion_batch, training=False)
    probs = _em_probs(emotion_output)
    idx = int(np.argmax(probs))
    smile_detected = detect_smile_bgr(face_bgr)
    if smile_detected and idx != HAPPY_INDEX:
        top_confidence = float(probs[idx])
        happy_confidence = float(probs[HAPPY_INDEX])
        if happy_confidence >= 0.08 or top_confidence < 0.95:
            idx = HAPPY_INDEX
            probs = probs.copy()
            probs[HAPPY_INDEX] = max(happy_confidence, 0.72)

    return FaceResult(
        gender=GENDER_LABELS[gender_idx],
        gender_confidence=float(gender_probs[gender_idx]),
        age=AGE_LABELS[age_idx],
        age_confidence=float(age_probs[age_idx]),
        emotion=EMOTION_LABELS[idx],
        emotion_confidence=float(probs[idx]),
    )


def predict_emotion_only(face_bgr: np.ndarray, models: Models) -> tuple[str, float]:
    """Predict only emotion for faster realtime updates.

    Uses direct model call to bypass predict_emotion() overhead.
    """
    from src.predict_emotion import (
        _preprocess_face as _preprocess_emotion,
        _to_probability_vector as _em_probs,
        EMOTION_LABELS, HAPPY_INDEX, detect_smile_bgr,
    )

    batch = _preprocess_emotion(face_bgr, IMAGE_SIZE)
    output = models.emotion_model(batch, training=False)
    probs = _em_probs(output)
    idx = int(np.argmax(probs))
    smile_detected = detect_smile_bgr(face_bgr)
    if smile_detected and idx != HAPPY_INDEX:
        top_confidence = float(probs[idx])
        happy_confidence = float(probs[HAPPY_INDEX])
        if happy_confidence >= 0.08 or top_confidence < 0.95:
            idx = HAPPY_INDEX
            probs = probs.copy()
            probs[HAPPY_INDEX] = max(happy_confidence, 0.72)
    return EMOTION_LABELS[idx], float(probs[idx])


def make_partial_face_result(emotion: str, confidence: float) -> FaceResult:
    """Create a temporary result while age/gender are still updating."""
    return FaceResult(
        gender="...",
        gender_confidence=0.0,
        age="...",
        age_confidence=0.0,
        emotion=emotion,
        emotion_confidence=confidence,
    )


def analyze_image_frame(
    frame: np.ndarray,
    models: Models,
    config: FaceDetectionConfig,
    *,
    max_faces: int | None = None,
    include_crops: bool = True,
) -> list[FacePrediction]:
    """Detect, predict, annotate, and return detailed face results."""
    original_frame = frame.copy()
    boxes = detect_faces_bgr(frame, config)
    if max_faces is not None:
        boxes = boxes[: max(0, int(max_faces))]

    predictions: list[FacePrediction] = []
    for box in boxes:
        try:
            face = crop_face_bgr(original_frame, box, margin=config.margin)
            result = predict_face(face, models)
        except Exception as exc:
            logger.warning("Skipping face with prediction error: %s", exc)
            continue

        draw_prediction(
            frame,
            box,
            age_label=result.age,
            gender_label=result.gender,
            emotion_label=result.emotion,
            score=result.emotion_confidence,
        )
        predictions.append(
            FacePrediction(
                box=box,
                result=result,
                crop_bgr=face.copy() if include_crops else None,
            )
        )
    return predictions


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw FPS text on a frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


class FaceDetectorGui:
    """Tkinter UI that keeps model inference off the main UI thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Age Gender Emotion Detector")
        self.root.geometry("1360x860")
        self.root.minsize(1060, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._maximize_window)

        self.models: Optional[Models] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.frame_queue: queue.Queue[Image.Image] = queue.Queue(maxsize=1)
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_size = DISPLAY_SIZE
        self.last_display_time = 0.0
        self.camera_view_active = False
        self.pending_image_open = False
        self.pending_snapshot_frame: np.ndarray | None = None
        self.latest_camera_frame: np.ndarray | None = None
        self.latest_camera_frame_lock = threading.Lock()
        self.latest_result: AnalysisResult | None = None
        self.history: list[AnalysisResult] = []
        self.result_photo_refs: list[ImageTk.PhotoImage] = []
        # Cache of live card widget-groups for in-place updates (avoids full destroy/recreate)
        self.face_cards: list[dict] = []

        self.status_var = tk.StringVar(value="Loading models...")
        self.camera_index_var = tk.StringVar(value="0")
        self.detector_backend_var = tk.StringVar(value="YOLO")
        self.quality_var = tk.StringVar(value="Quality")
        self.interval_ms_var = tk.DoubleVar(value=80)
        self.interval_label_var = tk.StringVar(value="80 ms")
        self.face_count_var = tk.StringVar(value="0")
        self.fps_var = tk.StringVar(value="--")
        self.latency_var = tk.StringVar(value="-- ms")
        self.active_detector_var = tk.StringVar(value="YOLO")
        self.result_state_var = tk.StringVar(value="Idle")

        # Game State variables
        self.current_mode = "detection"
        self.game_active = False
        self.game_target_emotion = None
        self.game_start_time = 0.0
        self.game_duration = 5.0
        self.game_max_score = 0.0
        self.game_current_score = 0.0
        self.game_stage = "idle"

        self.game_status_var = tk.StringVar(value="Đang chờ bắt đầu...")
        self.game_target_var = tk.StringVar(value="--")
        self.game_score_text_var = tk.StringVar(value="0%")
        self.game_max_score_text_var = tk.StringVar(value="Cao nhất: 0%")
        self.game_feedback_var = tk.StringVar(value="Nhấn 'Bắt đầu chơi' để chơi.")
        self._build_ui()
        self._set_controls_enabled(False)
        self._start_model_loader()
        self._pump_frames()

    def _build_ui(self) -> None:
        self._configure_styles()
        self.root.configure(bg="#0b0f19")

        self.shell = ttk.Frame(self.root, style="App.TFrame", padding=(14, 12, 14, 14))
        self.shell.pack(fill=tk.BOTH, expand=True)

        self.header = ttk.Frame(self.shell, style="App.TFrame")
        self.header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(self.header, text="Face Analysis", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(self.header, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.RIGHT
        )

        self.workspace = ttk.Frame(self.shell, style="App.TFrame")
        self.workspace.pack(fill=tk.BOTH, expand=True)

        self.main_panel = ttk.Frame(self.workspace, style="App.TFrame")
        self.main_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.side_panel = ttk.Frame(
            self.workspace,
            style="Panel.TFrame",
            width=380,
            padding=(14, 14, 14, 14),
        )
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        self.side_panel.pack_propagate(False)

        preview_frame = tk.Frame(
            self.main_panel,
            bg="#020617",
            highlightbackground="#334155",
            highlightthickness=1,
        )
        preview_frame.pack(fill=tk.BOTH, expand=True)
        preview_frame.pack_propagate(False)
        self.preview_frame = preview_frame

        self.preview_label = tk.Label(
            preview_frame,
            bg="#020617",
            fg="#e5e7eb",
            text="Loading models...",
            anchor=tk.CENTER,
            font=("Segoe UI", 14),
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_label.bind("<Configure>", self._update_preview_size)

        self.metrics = ttk.Frame(self.main_panel, style="Panel.TFrame", padding=(0, 10, 0, 0))
        self.metrics.pack(fill=tk.X)
        self._add_metric(self.metrics, "Faces", self.face_count_var, 0)
        self._add_metric(self.metrics, "FPS", self.fps_var, 1)
        self._add_metric(self.metrics, "Latency", self.latency_var, 2)
        self._add_metric(self.metrics, "Detector", self.active_detector_var, 3)

        self._build_side_panel_content(self.side_panel)

        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._exit_fullscreen)
        self.root.bind("<Return>", self._stop_from_key)
        self.root.bind("<space>", self._stop_from_key)

    def _add_metric(self, parent, title: str, variable: tk.StringVar, column: int) -> None:
        parent.columnconfigure(column, weight=1, uniform="metrics")
        cell = ttk.Frame(parent, style="Metric.TFrame", padding=(12, 10, 12, 10))
        cell.grid(row=0, column=column, sticky=tk.EW, padx=(0 if column == 0 else 8, 0))
        ttk.Label(cell, text=title, style="MetricTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(cell, textvariable=variable, style="MetricValue.TLabel").pack(anchor=tk.W)

    def _build_side_panel_content(self, parent) -> None:
        """Build side panel with Mode Switcher and container frames."""
        # 1. Mode Switcher buttons
        self.mode_switcher = ttk.Frame(parent, style="Panel.TFrame")
        self.mode_switcher.pack(fill=tk.X, pady=(0, 14))
        self.mode_switcher.columnconfigure(0, weight=1)
        self.mode_switcher.columnconfigure(1, weight=1)
        
        self.mode_detection_btn = ttk.Button(
            self.mode_switcher,
            text="Nhận diện",
            command=lambda: self._switch_mode("detection"),
            style="Primary.TButton"
        )
        self.mode_detection_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4))
        
        self.mode_game_btn = ttk.Button(
            self.mode_switcher,
            text="Trò chơi",
            command=lambda: self._switch_mode("game"),
            style="Tool.TButton"
        )
        self.mode_game_btn.grid(row=0, column=1, sticky=tk.EW, padx=(4, 0))
        
        # 2. Side content container
        self.side_content_frame = ttk.Frame(parent, style="Panel.TFrame")
        self.side_content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 3. Create Detection View Frame
        self.detection_view_frame = ttk.Frame(self.side_content_frame, style="Panel.TFrame")
        self._build_controls(self.detection_view_frame)
        
        self.history_container = ttk.Frame(self.detection_view_frame, style="Panel.TFrame")
        self.history_container.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_history(self.history_container)
        
        self._build_results(self.detection_view_frame)
        self.detection_view_frame.pack(fill=tk.BOTH, expand=True)
        
        # 4. Create Game View Frame
        self.game_view_frame = ttk.Frame(self.side_content_frame, style="Panel.TFrame")
        self._build_game_ui(self.game_view_frame)

    def _build_controls(self, parent) -> None:
        ttk.Label(parent, text="Controls", style="Section.TLabel").pack(anchor=tk.W)

        camera_row = ttk.Frame(parent, style="Panel.TFrame")
        camera_row.pack(fill=tk.X, pady=(10, 8))
        ttk.Label(camera_row, text="Camera", style="Body.TLabel").pack(side=tk.LEFT)
        self.camera_entry = ttk.Entry(
            camera_row,
            width=7,
            textvariable=self.camera_index_var,
            justify=tk.CENTER,
        )
        self.camera_entry.pack(side=tk.RIGHT, ipady=3)

        self.detector_combo = self._control_combo(
            parent,
            "Detector",
            self.detector_backend_var,
            ("YOLO", "Auto", "Haar"),
        )

        self.webcam_button = ttk.Button(
            parent,
            text="Start Camera",
            command=self.start_webcam,
            style="Primary.TButton",
        )
        self.webcam_button.pack(fill=tk.X, pady=(2, 8))

        action_grid = ttk.Frame(parent, style="Panel.TFrame")
        action_grid.pack(fill=tk.X)
        action_grid.columnconfigure(0, weight=1)
        action_grid.columnconfigure(1, weight=1)
        self.snapshot_button = ttk.Button(
            action_grid,
            text="Snapshot",
            command=self.snapshot_camera,
            style="Tool.TButton",
        )
        self.snapshot_button.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5), pady=(0, 8))
        self.stop_button = ttk.Button(
            action_grid,
            text="Stop",
            command=self.stop_current,
            style="Danger.TButton",
        )
        self.stop_button.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0), pady=(0, 8))

        self.image_button = ttk.Button(
            action_grid,
            text="Open Image",
            command=self.choose_image,
            style="Tool.TButton",
        )
        self.image_button.grid(row=1, column=0, sticky=tk.EW, padx=(0, 5))
        self.save_button = ttk.Button(
            action_grid,
            text="Save Result",
            command=self.save_current_result,
            style="Tool.TButton",
        )
        self.save_button.grid(row=1, column=1, sticky=tk.EW, padx=(5, 0))

        self.quit_button = ttk.Button(
            parent,
            text="Exit",
            command=self.close,
            style="Tool.TButton",
        )
        self.quit_button.pack(fill=tk.X, pady=(12, 0))

    def _control_combo(
        self,
        parent,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label, style="Body.TLabel").pack(anchor=tk.W, pady=(8, 4))
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            justify=tk.CENTER,
        )
        combo.pack(fill=tk.X, ipady=3)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_active_detector())
        return combo

    def _build_results(self, parent) -> None:
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.pack(fill=tk.X, pady=(18, 8))
        ttk.Label(header, text="Results", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.result_state_var, style="Muted.TLabel").pack(
            side=tk.RIGHT
        )

        self.results_scroll = ScrollableFrame(parent, height=200)
        self.results_scroll.pack(fill=tk.BOTH, expand=True)
        self._render_predictions([])

    def _build_history(self, parent) -> None:
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.pack(fill=tk.X, pady=(18, 8))
        ttk.Label(header, text="History", style="Section.TLabel").pack(side=tk.LEFT)
        self.clear_history_button = ttk.Button(
            header,
            text="Clear",
            command=self.clear_history,
            style="Small.TButton",
        )
        self.clear_history_button.pack(side=tk.RIGHT)

        self.history_listbox = tk.Listbox(
            parent,
            height=7,
            bg="#0f172a",
            fg="#e5e7eb",
            selectbackground="#2563eb",
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#334155",
            relief=tk.FLAT,
            activestyle="none",
            font=("Segoe UI", 9),
        )
        self.history_listbox.pack(fill=tk.X)
        self.history_listbox.bind("<<ListboxSelect>>", self._on_history_select)
        self._render_history()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Sleek Modern Dark Palette
        style.configure("App.TFrame", background="#0b0f19")
        style.configure("Panel.TFrame", background="#161f30")
        style.configure("Metric.TFrame", background="#161f30")
        style.configure("Card.TFrame", background="#1e293b")
        style.configure(
            "Title.TLabel",
            background="#0b0f19",
            foreground="#f8fafc",
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background="#0b0f19",
            foreground="#60a5fa",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background="#161f30",
            foreground="#f8fafc",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#161f30",
            foreground="#cbd5e1",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background="#161f30",
            foreground="#94a3b8",
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardTitle.TLabel",
            background="#1e293b",
            foreground="#f8fafc",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "CardBody.TLabel",
            background="#1e293b",
            foreground="#cbd5e1",
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardMuted.TLabel",
            background="#1e293b",
            foreground="#94a3b8",
            font=("Segoe UI", 8),
        )
        style.configure(
            "MetricTitle.TLabel",
            background="#161f30",
            foreground="#94a3b8",
            font=("Segoe UI", 8, "bold"),
        )
        style.configure(
            "MetricValue.TLabel",
            background="#161f30",
            foreground="#f8fafc",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Primary.TButton",
            background="#3b82f6",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10, "bold"),
            padding=(16, 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#2563eb"), ("disabled", "#334155")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure(
            "Tool.TButton",
            background="#334155",
            foreground="#f8fafc",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10),
            padding=(14, 9),
        )
        style.map(
            "Tool.TButton",
            background=[("active", "#475569"), ("disabled", "#1e293b")],
            foreground=[("disabled", "#64748b")],
        )
        style.configure(
            "Danger.TButton",
            background="#ef4444",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10, "bold"),
            padding=(14, 9),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#dc2626"), ("disabled", "#334155")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure(
            "Small.TButton",
            background="#334155",
            foreground="#f8fafc",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 8),
            padding=(8, 4),
        )
        style.configure("Result.Horizontal.TProgressbar", troughcolor="#334155", background="#10b981")
        style.configure(
            "TScrollbar",
            troughcolor="#161f30",
            background="#334155",
            arrowcolor="#f8fafc",
            bordercolor="#161f30",
            darkcolor="#161f30",
            lightcolor="#161f30",
        )

    def _start_model_loader(self) -> None:
        thread = threading.Thread(target=self._load_models_worker, daemon=True)
        thread.start()

    def _load_models_worker(self) -> None:
        try:
            models = load_models()
        except Exception as exc:
            logger.exception("Could not load models")
            details = traceback.format_exc()
            self._run_on_ui(
                self._model_load_failed,
                f"Could not load models:\n\n{exc}\n\n{details}",
            )
            return

        self.models = models
        self._run_on_ui(self._model_load_succeeded)

    def _model_load_succeeded(self) -> None:
        self.status_var.set("Ready")
        self.preview_label.configure(text="Select a source to start")
        self._set_controls_enabled(True)

    def _model_load_failed(self, message: str) -> None:
        self.status_var.set(message)
        self.preview_label.configure(text=message)
        messagebox.showerror("Model Error", message)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (
            self.camera_entry,
            self.webcam_button,
            self.image_button,
        ):
            widget.configure(state=state)
        self.detector_combo.configure(state="readonly" if enabled else tk.DISABLED)
        self.snapshot_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)
        self._set_save_enabled(enabled and self.latest_result is not None)
        self.quit_button.configure(state=tk.NORMAL)

    def _set_camera_running(self) -> None:
        for widget in (
            self.camera_entry,
            self.webcam_button,
            self.detector_combo,
        ):
            widget.configure(state=tk.DISABLED)
        self.image_button.configure(state=tk.NORMAL)
        self.snapshot_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL)
        self.save_button.configure(state=tk.DISABLED)
        self.quit_button.configure(state=tk.NORMAL)

    def _set_processing_running(self) -> None:
        for widget in (
            self.camera_entry,
            self.webcam_button,
            self.image_button,
            self.detector_combo,
            self.snapshot_button,
        ):
            widget.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.quit_button.configure(state=tk.NORMAL)

    def _set_idle_controls(self) -> None:
        if self.models is None:
            self._set_controls_enabled(False)
            return
        self.camera_entry.configure(state=tk.NORMAL)
        self.webcam_button.configure(state=tk.NORMAL)
        self.image_button.configure(state=tk.NORMAL)
        self.detector_combo.configure(state="readonly")
        self.snapshot_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)
        self.stop_button.configure(text="Stop")
        self._set_save_enabled(self.latest_result is not None)
        self.quit_button.configure(state=tk.NORMAL)

    def _set_save_enabled(self, enabled: bool) -> None:
        self.save_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def start_webcam(self) -> None:
        if self.models is None or self._is_worker_alive():
            return

        try:
            camera_index = int(self.camera_index_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror("Invalid Camera Index", "Camera index must be an integer.")
            return

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Camera Error", f"Could not open camera index {camera_index}.")
            return

        config = self._face_detection_config(source="webcam")
        self.status_var.set("Camera running")
        self.result_state_var.set("Live")
        self._enter_camera_view()
        self._start_capture_worker(cap, config)

    def snapshot_camera(self) -> None:
        if not self.camera_view_active or self.stop_event is None:
            return
        frame = self._copy_latest_camera_frame()
        if frame is None:
            self.status_var.set("No camera frame available yet")
            return
        self.pending_snapshot_frame = frame
        self.status_var.set("Capturing snapshot")
        self.stop_current()

    def choose_image(self) -> None:
        if self.camera_view_active and self.stop_event is not None:
            self.pending_image_open = True
            self.stop_current()
            return
        self._choose_image_now()

    def _choose_image_now(self) -> None:
        if self.models is None or self._is_worker_alive():
            return

        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        config = self._face_detection_config(source="image")
        self.status_var.set("Processing image")
        self.result_state_var.set("Analyzing")
        self._set_processing_running()
        self.worker_thread = threading.Thread(
            target=self._image_file_worker,
            args=(
                Path(file_path),
                config,
                self._active_detector_label(),
                self.quality_var.get(),
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def save_current_result(self) -> None:
        if self.latest_result is None:
            return

        initial = f"face-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
        path = filedialog.asksaveasfilename(
            title="Save Result",
            initialfile=initial,
            defaultextension=".jpg",
            filetypes=[
                ("JPEG image", "*.jpg"),
                ("PNG image", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        if not cv2.imwrite(str(path), self.latest_result.frame_bgr):
            messagebox.showerror("Save Error", f"Could not save result to:\n{path}")
            return
        self.latest_result.saved_path = Path(path)
        self.status_var.set(f"Saved result: {path}")
        self._render_history()

    def _start_capture_worker(
        self,
        cap: cv2.VideoCapture,
        config: FaceDetectionConfig,
    ) -> None:
        self.stop_event = threading.Event()
        self._set_camera_running()
        self.stop_button.configure(text="Stop Camera")
        interval_seconds = self._realtime_interval_seconds()
        self.worker_thread = threading.Thread(
            target=self._capture_worker,
            args=(cap, self.stop_event, config, interval_seconds),
            daemon=True,
        )
        self.worker_thread.start()

    def _capture_worker(
        self,
        cap: cv2.VideoCapture,
        stop_event: threading.Event,
        config: FaceDetectionConfig,
        interval_seconds: float,
    ) -> None:
        processor = AsyncRealtimeFaceProcessor(
            self.models,
            predict_face,
            config,
            predict_emotion=predict_emotion_only,
            partial_result_factory=make_partial_face_result,
            inference_interval=interval_seconds,
            full_inference_interval=0.35,
            detection_width=320,
            max_faces=3,
            emotion_history_size=2,
            emotion_min_confidence=0.25,
            label_switch_margin=0.08,
            label_switch_count=2,
        )
        prev_time = time.time()
        last_ui_update = 0.0

        try:
            processor.start()
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                self._store_latest_camera_frame(frame)
                started = time.perf_counter()
                processor.process_frame(frame)
                draw_ms = int((time.perf_counter() - started) * 1000)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                draw_fps(frame, fps)

                if now - last_ui_update >= 0.25:
                    cached = processor._get_cached_predictions()
                    predictions = [
                        FacePrediction(box=item.box, result=copy(item.result))
                        for item in cached
                    ]
                    self._run_on_ui(self._update_live_results, predictions, fps, draw_ms)
                    last_ui_update = now

                if self.game_active:
                    self._draw_game_overlay(frame)

                self._queue_frame(frame)
                time.sleep(0.001)
        finally:
            processor.stop()
            cap.release()

        if stop_event.is_set():
            message = "Stopped"
        else:
            message = "Camera ended"
        self._run_on_ui(self._finish_camera_worker, message)

    def _image_file_worker(
        self,
        path: Path,
        config: FaceDetectionConfig,
        detector: str,
        quality: str,
    ) -> None:
        frame = cv2.imread(str(path))
        if frame is None:
            self._run_on_ui(self._finish_analysis_error, f"Could not read image: {path}")
            return

        self._analyze_frame_worker(
            frame,
            source=path.name,
            config=config,
            detector=detector,
            quality=quality,
            original_path=path,
        )

    def _snapshot_worker(
        self,
        frame: np.ndarray,
        config: FaceDetectionConfig,
        detector: str,
        quality: str,
    ) -> None:
        self._analyze_frame_worker(
            frame,
            source="Snapshot",
            config=config,
            detector=detector,
            quality=quality,
            original_path=None,
        )

    def _analyze_frame_worker(
        self,
        frame: np.ndarray,
        *,
        source: str,
        config: FaceDetectionConfig,
        detector: str,
        quality: str,
        original_path: Path | None,
    ) -> None:
        started = time.perf_counter()
        predictions = analyze_image_frame(
            frame,
            self.models,
            config,
            max_faces=config.yolo_max_det,
            include_crops=True,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        saved_path = None

        height, width = frame.shape[:2]
        result = AnalysisResult(
            source=source,
            frame_bgr=frame,
            frame_width=width,
            frame_height=height,
            face_count=len(predictions),
            elapsed_ms=elapsed_ms,
            detector=detector,
            quality=quality,
            predictions=predictions,
            saved_path=saved_path,
        )
        self._queue_frame(frame, force=True)
        self._run_on_ui(self._finish_analysis_worker, result)

    def _finish_camera_worker(self, message: str) -> None:
        self.status_var.set(message)
        self.stop_event = None
        self.worker_thread = None
        self.face_cards.clear()          # discard stale card refs from live session
        self.result_photo_refs.clear()
        self._exit_camera_view()
        self._set_idle_controls()

        if self.pending_snapshot_frame is not None:
            frame = self.pending_snapshot_frame
            self.pending_snapshot_frame = None
            self.root.after(120, lambda: self._start_snapshot_analysis(frame))
            return

        if self.pending_image_open:
            self.pending_image_open = False
            self.root.after(150, self._choose_image_now)

    def _start_snapshot_analysis(self, frame: np.ndarray) -> None:
        if self.models is None or self._is_worker_alive():
            return
        config = self._face_detection_config(source="image")
        detector = self._active_detector_label()
        quality = self.quality_var.get()
        self.status_var.set("Processing snapshot")
        self.result_state_var.set("Analyzing")
        self._set_processing_running()
        self.worker_thread = threading.Thread(
            target=self._snapshot_worker,
            args=(frame, config, detector, quality),
            daemon=True,
        )
        self.worker_thread.start()

    def _finish_analysis_worker(self, result: AnalysisResult) -> None:
        self.worker_thread = None
        self._show_analysis_result(result, save_history=True)
        if result.face_count == 0:
            message = "No faces detected"
        elif result.source == "Snapshot":
            message = "Snapshot ready"
        else:
            message = "Analysis complete"
        self.status_var.set(message)
        self._set_idle_controls()

    def _finish_analysis_error(self, message: str) -> None:
        self.worker_thread = None
        self.status_var.set(message)
        self.result_state_var.set("Error")
        self._set_idle_controls()
        messagebox.showerror("Image Error", message)

    def _show_analysis_result(self, result: AnalysisResult, *, save_history: bool) -> None:
        self.latest_result = result
        self.face_count_var.set(str(result.face_count))
        self.fps_var.set("--")
        self.latency_var.set(f"{result.elapsed_ms} ms")
        self.active_detector_var.set(result.detector)
        self.result_state_var.set("Ready" if result.face_count else "No faces")
        self._queue_frame(result.frame_bgr, force=True)
        self._render_predictions(result.predictions)
        self._set_save_enabled(True)
        if save_history:
            self._add_history(result)

    def _update_live_results(
        self,
        predictions: list[FacePrediction],
        fps: float,
        draw_ms: int,
    ) -> None:
        if not self.camera_view_active:
            return
        self.face_count_var.set(str(len(predictions)))
        self.fps_var.set(f"{fps:.1f}")
        self.latency_var.set(f"{draw_ms} ms")
        self.active_detector_var.set(self._active_detector_label())
        self.result_state_var.set("Live")
        self._render_predictions(predictions)
        
        # Game Mode scoring updates
        if self.game_active and self.game_stage == "playing":
            if predictions and self.game_target_emotion:
                main_face = predictions[0]
                result = main_face.result
                if result.emotion.lower() == self.game_target_emotion.lower():
                    score = float(result.emotion_confidence) * 100
                else:
                    score = 0.0
            else:
                score = 0.0
                
            self.game_current_score = score
            self.game_score_bar["value"] = int(score)
            self.game_score_text_var.set(f"{int(score)}%")
            
            if score > self.game_max_score:
                self.game_max_score = score
                self.game_max_score_text_var.set(f"Cao nhất: {int(score)}%")
                
                # Dynamic Vietnamese feedback
                if score >= 85:
                    self.game_feedback_var.set("Tuyệt vời! Rất giống! 🔥")
                elif score >= 60:
                    self.game_feedback_var.set("Khá lắm, giữ nét mặt nhé! 👍")
                elif score >= 35:
                    self.game_feedback_var.set("Hơi giống rồi đó! 🙂")
                else:
                    self.game_feedback_var.set("Hãy biểu cảm rõ hơn chút! 💪")

    def stop_current(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("Stopping")
            self.stop_button.configure(state=tk.DISABLED)
            self.snapshot_button.configure(state=tk.DISABLED)

    def close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self.root.after(100, self.root.destroy)

    def clear_history(self) -> None:
        self.history = []
        self._render_history()

    def _add_history(self, result: AnalysisResult) -> None:
        self.history.insert(0, result)
        self.history = self.history[:HISTORY_LIMIT]
        self._render_history()

    def _render_history(self) -> None:
        self.history_listbox.delete(0, tk.END)
        if not self.history:
            self.history_listbox.insert(tk.END, "No saved results")
            self.history_listbox.itemconfigure(0, foreground="#94a3b8")
            return

        for item in self.history:
            emotion = item.predictions[0].result.emotion if item.predictions else "No face"
            path_marker = " saved" if item.saved_path else ""
            label = (
                f"{item.created_at}  {item.source}  "
                f"{item.face_count} face(s)  {emotion}  {item.detector}{path_marker}"
            )
            self.history_listbox.insert(tk.END, label)

    def _on_history_select(self, _event=None) -> None:
        if self.camera_view_active or not self.history:
            return
        selection = self.history_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index >= len(self.history):
            return
        result = self.history[index]
        self._show_analysis_result(result, save_history=False)
        self.status_var.set(f"History: {result.source}")

    def _render_predictions(self, predictions: list[FacePrediction]) -> None:
        # In game mode: skip the sidebar face-card panel entirely to reduce UI thread work
        if self.current_mode == "game":
            return

        current_count = len(predictions)
        cached_count = len(self.face_cards)

        # If face count changed, do a full rebuild (rare event)
        if current_count != cached_count:
            for child in self.results_scroll.content.winfo_children():
                child.destroy()
            self.result_photo_refs.clear()
            self.face_cards.clear()

            if not predictions:
                ttk.Label(
                    self.results_scroll.content,
                    text="No face details",
                    style="Muted.TLabel",
                    padding=(12, 14),
                ).pack(fill=tk.X)
                return

            for index, prediction in enumerate(predictions, start=1):
                card_refs = self._add_prediction_card(index, prediction)
                self.face_cards.append(card_refs)
            return

        # Same face count: update widgets in-place (no destroy/recreate)
        for card_refs, prediction in zip(self.face_cards, predictions):
            result = prediction.result
            self._update_prediction_card(card_refs, prediction, result)

    def _add_prediction_card(self, index: int, prediction: FacePrediction) -> dict:
        """Build card widgets and return references for in-place updates."""
        result = prediction.result
        card = ttk.Frame(self.results_scroll.content, style="Card.TFrame", padding=(10, 10, 10, 10))
        card.pack(fill=tk.X, pady=(0, 10))
        card.columnconfigure(1, weight=1)

        if prediction.crop_bgr is not None:
            photo = self._make_crop_photo(prediction.crop_bgr)
            self.result_photo_refs.append(photo)
            crop_label = ttk.Label(card, image=photo, style="CardBody.TLabel")
        else:
            photo = None
            crop_label = tk.Label(
                card,
                text=f"Face {index}",
                width=10,
                height=5,
                bg="#0f172a",
                fg="#94a3b8",
                font=("Segoe UI", 9, "bold"),
            )
        crop_label.grid(row=0, column=0, rowspan=5, sticky=tk.NW, padx=(0, 10))

        title_label = ttk.Label(
            card,
            text=f"Face {index} - {result.emotion}",
            style="CardTitle.TLabel",
        )
        title_label.grid(row=0, column=1, sticky=tk.EW)

        e_refs = self._add_confidence_row(card, 1, "Emotion", result.emotion_confidence, result.emotion)
        g_refs = self._add_confidence_row(card, 2, "Gender", result.gender_confidence, result.gender)
        a_refs = self._add_confidence_row(card, 3, "Age", result.age_confidence, result.age)

        x, y, w, h = prediction.box
        box_label = ttk.Label(
            card,
            text=f"Box {w}x{h} at {x}, {y}",
            style="CardMuted.TLabel",
        )
        box_label.grid(row=4, column=1, sticky=tk.EW, pady=(4, 0))

        return {
            "crop_label": crop_label,
            "title_label": title_label,
            "box_label": box_label,
            "emotion": e_refs,
            "gender": g_refs,
            "age": a_refs,
            "index": index,
        }

    def _update_prediction_card(self, card_refs: dict, prediction: FacePrediction, result) -> None:
        """Update existing card widgets in-place — no widget creation overhead."""
        index = card_refs["index"]
        card_refs["title_label"].configure(text=f"Face {index} - {result.emotion}")

        if prediction.crop_bgr is not None:
            photo = self._make_crop_photo(prediction.crop_bgr)
            self.result_photo_refs.append(photo)
            card_refs["crop_label"].configure(image=photo)

        x, y, w, h = prediction.box
        card_refs["box_label"].configure(text=f"Box {w}x{h} at {x}, {y}")

        self._update_confidence_row(card_refs["emotion"], result.emotion_confidence, result.emotion)
        self._update_confidence_row(card_refs["gender"], result.gender_confidence, result.gender)
        self._update_confidence_row(card_refs["age"], result.age_confidence, result.age)

    def _add_confidence_row(
        self,
        parent,
        row: int,
        label: str,
        confidence: float,
        value: str,
    ) -> dict:
        """Build a confidence row and return refs for in-place updates."""
        percent = max(0, min(100, int(round(float(confidence) * 100))))
        wrapper = ttk.Frame(parent, style="Card.TFrame")
        wrapper.grid(row=row, column=1, sticky=tk.EW, pady=(6, 0))
        wrapper.columnconfigure(1, weight=1)
        ttk.Label(wrapper, text=label, style="CardBody.TLabel", width=8).grid(
            row=0, column=0, sticky=tk.W,
        )
        bar = ttk.Progressbar(
            wrapper,
            maximum=100,
            value=percent,
            style="Result.Horizontal.TProgressbar",
        )
        bar.grid(row=0, column=1, sticky=tk.EW, padx=(8, 8))
        pct_label = ttk.Label(wrapper, text=f"{percent}%", style="CardBody.TLabel", width=5)
        pct_label.grid(row=0, column=2, sticky=tk.E)
        val_label = ttk.Label(wrapper, text=value, style="CardMuted.TLabel")
        val_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))
        return {"bar": bar, "pct_label": pct_label, "val_label": val_label}

    def _update_confidence_row(self, refs: dict, confidence: float, value: str) -> None:
        """Update confidence bar, percent text and value label in-place."""
        percent = max(0, min(100, int(round(float(confidence) * 100))))
        refs["bar"]["value"] = percent
        refs["pct_label"].configure(text=f"{percent}%")
        refs["val_label"].configure(text=value)

    def _make_crop_photo(self, crop_bgr: np.ndarray) -> ImageTk.PhotoImage:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        image = ImageOps.fit(image, (78, 78), method=Image.Resampling.BILINEAR)
        return ImageTk.PhotoImage(image=image)

    def _queue_frame(self, frame_bgr: np.ndarray, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_display_time < DISPLAY_INTERVAL:
            return
        self.last_display_time = now

        # Use cv2.resize instead of PIL for ~3x faster frame conversion
        target_w, target_h = self.preview_size
        h, w = frame_bgr.shape[:2]
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(image)
        except queue.Full:
            pass

    def _pump_frames(self) -> None:
        try:
            image = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.current_photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self.current_photo, text="")
        self.root.after(15, self._pump_frames)

    def _update_preview_size(self, event) -> None:
        width = max(320, int(event.width))
        height = max(240, int(event.height))
        self.preview_size = (width, height)

    def _maximize_window(self) -> None:
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

    def _toggle_fullscreen(self, _event=None) -> None:
        enabled = not bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", enabled)

    def _exit_fullscreen(self, _event=None) -> None:
        self.root.attributes("-fullscreen", False)

    def _enter_camera_view(self) -> None:
        self.camera_view_active = True
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

    def _exit_camera_view(self) -> None:
        if not self.camera_view_active:
            return
        self.camera_view_active = False
        self.root.attributes("-fullscreen", False)

    def _stop_from_key(self, _event=None) -> None:
        if self.camera_view_active:
            self.stop_current()

    def _is_worker_alive(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _run_on_ui(self, func, *args) -> None:
        try:
            self.root.after(0, func, *args)
        except tk.TclError:
            pass

    def _store_latest_camera_frame(self, frame_bgr: np.ndarray) -> None:
        with self.latest_camera_frame_lock:
            self.latest_camera_frame = frame_bgr.copy()

    def _copy_latest_camera_frame(self) -> np.ndarray | None:
        with self.latest_camera_frame_lock:
            if self.latest_camera_frame is None:
                return None
            return self.latest_camera_frame.copy()

    def _sync_active_detector(self) -> None:
        self.active_detector_var.set(self._active_detector_label())

    def _active_detector_label(self) -> str:
        return self.detector_backend_var.get().strip().upper() or "YOLO"

    def _realtime_interval_seconds(self) -> float:
        return max(0.05, float(self.interval_ms_var.get()) / 1000.0)

    def _on_interval_change(self, value: str) -> None:
        interval = int(round(float(value)))
        self.interval_ms_var.set(interval)
        self.interval_label_var.set(f"{interval} ms")

    def _face_detection_config(self, *, source: str) -> FaceDetectionConfig:
        backend = self.detector_backend_var.get().strip().lower() or "yolo"

        if backend not in {"yolo", "auto", "haar"}:
            backend = "yolo"

        if source == "image":
            return FaceDetectionConfig(
                backend=backend,
                min_neighbors=4,
                min_size=(24, 24),
                margin=0.35,
                yolo_imgsz=640,
                yolo_confidence=0.28,
                yolo_iou=0.45,
                yolo_max_det=20,
            )

        return FaceDetectionConfig(
            backend=backend,
            min_neighbors=4,
            min_size=(32, 32),
            margin=0.35,
            yolo_imgsz=320,
            yolo_max_det=20,
        )

    def _build_game_ui(self, parent) -> None:
        """Create UI elements for the Emotion Mimic Game."""

        ttk.Label(parent, text="Trò chơi Bắt chước", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        # Instructions Box
        instr_card = ttk.Frame(parent, style="Card.TFrame", padding=(12, 10, 12, 10))
        instr_card.pack(fill=tk.X, pady=(0, 14))
        
        ttk.Label(
            instr_card,
            text="Hệ thống sẽ đưa ra một cảm xúc ngẫu nhiên. Hãy bắt chước cảm xúc đó trước camera để đạt điểm tối đa!",
            style="CardBody.TLabel",
            wraplength=320,
            justify=tk.LEFT
        ).pack(anchor=tk.W)
        
        # Target Display Panel
        target_card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 14, 14, 14))
        target_card.pack(fill=tk.X, pady=(0, 14))
        
        ttk.Label(target_card, text="MỤC TIÊU CẦN BIỂU CẢM:", style="CardMuted.TLabel").pack(anchor=tk.CENTER)
        self.game_target_label = ttk.Label(
            target_card,
            textvariable=self.game_target_var,
            font=("Segoe UI", 20, "bold"),
            foreground="#f59e0b",
            background="#1e293b"
        )
        self.game_target_label.pack(anchor=tk.CENTER, pady=(6, 0))
        
        # Current Score Progress
        score_card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12, 14, 12))
        score_card.pack(fill=tk.X, pady=(0, 14))
        
        score_row = ttk.Frame(score_card, style="Card.TFrame")
        score_row.pack(fill=tk.X)
        ttk.Label(score_row, text="Độ khớp nét mặt:", style="CardBody.TLabel").pack(side=tk.LEFT)
        ttk.Label(score_row, textvariable=self.game_score_text_var, style="CardTitle.TLabel").pack(side=tk.RIGHT)
        
        self.game_score_bar = ttk.Progressbar(
            score_card,
            maximum=100,
            value=0,
            style="Result.Horizontal.TProgressbar"
        )
        self.game_score_bar.pack(fill=tk.X, pady=(6, 8))
        
        ttk.Label(score_card, textvariable=self.game_max_score_text_var, style="CardMuted.TLabel").pack(anchor=tk.W)
        
        # Status / Feedback Box
        status_card = ttk.Frame(parent, style="Card.TFrame", padding=(12, 12, 12, 12))
        status_card.pack(fill=tk.X, pady=(0, 18))
        
        ttk.Label(status_card, text="Trạng thái:", style="CardMuted.TLabel").pack(anchor=tk.W)
        ttk.Label(
            status_card,
            textvariable=self.game_status_var,
            style="CardTitle.TLabel",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor=tk.W, pady=(4, 2))
        
        ttk.Label(
            status_card,
            textvariable=self.game_feedback_var,
            style="CardBody.TLabel",
            font=("Segoe UI", 9, "italic")
        ).pack(anchor=tk.W)
        
        # Control Buttons
        self.game_start_btn = ttk.Button(
            parent,
            text="Bắt đầu chơi 🎮",
            command=self._start_game,
            style="Primary.TButton"
        )
        self.game_start_btn.pack(fill=tk.X, pady=(0, 8))
        
        self.game_stop_btn = ttk.Button(
            parent,
            text="Dừng chơi",
            command=self._stop_game,
            style="Danger.TButton"
        )
        self.game_stop_btn.pack(fill=tk.X)
        self.game_stop_btn.configure(state=tk.DISABLED)

    def _start_game(self) -> None:
        """Initialize the game round and start the camera if needed."""

        if self.models is None:
            messagebox.showerror("Lỗi", "Mô hình chưa tải xong, vui lòng đợi một chút.")
            return

        if not self.camera_view_active:
            self.start_webcam()
            # If camera still failed to open
            if not self.camera_view_active:
                messagebox.showerror("Lỗi", "Hãy kết nối và bật camera trước khi chơi!")
                return
        
        import random
        GAME_EMOTIONS = ["Happy", "Sad", "Surprise", "Anger"]
        target = random.choice(GAME_EMOTIONS)
        
        EMOTION_TRANSLATE = {
            "Happy": "Hạnh phúc 😊",
            "Sad": "Buồn bã 😢",
            "Surprise": "Ngạc nhiên 😲",
            "Anger": "Tức giận 😠"
        }
        
        self.game_target_emotion = target
        self.game_target_var.set(EMOTION_TRANSLATE[target])
        self.game_stage = "countdown"
        self.game_start_time = time.time()
        self.game_max_score = 0.0
        self.game_current_score = 0.0
        self.game_active = True
        
        self.game_start_btn.configure(state=tk.DISABLED)
        self.game_stop_btn.configure(state=tk.NORMAL)
        
        self.game_status_var.set("Chuẩn bị...")
        self.game_feedback_var.set("Chuẩn bị biểu diễn nét mặt!")
        self.game_score_bar["value"] = 0
        self.game_score_text_var.set("0%")
        self.game_max_score_text_var.set("Cao nhất: 0%")
        
        self._game_tick()

    def _game_tick(self) -> None:
        """Game timer tick update loop running on the UI thread."""

        if not self.game_active:
            return
            
        now = time.time()
        elapsed = now - self.game_start_time
        
        if self.game_stage == "countdown":
            remaining = 3.0 - elapsed
            if remaining <= 0:
                self.game_stage = "playing"
                self.game_start_time = time.time()
                self.game_max_score = 0.0
                self.game_status_var.set("BẮT CHƯỚC NGAY!")
                self.game_feedback_var.set("Biểu cảm khuôn mặt trước camera!")
            else:
                self.game_status_var.set(f"Sắp bắt đầu... {int(remaining) + 1}s")
                self.game_feedback_var.set("Hãy nhìn thẳng vào ống kính.")
                
        elif self.game_stage == "playing":
            remaining = self.game_duration - elapsed
            if remaining <= 0:
                self.game_stage = "result"
                self.game_active = False
                self._finish_game_round()
                return
            else:
                self.game_status_var.set(f"Thời gian còn lại: {remaining:.1f}s")
                
        self.root.after(50, self._game_tick)

    def _finish_game_round(self) -> None:
        """Finalize game round, display results, and reset buttons."""

        self.game_active = False
        self.game_start_btn.configure(state=tk.NORMAL)
        self.game_stop_btn.configure(state=tk.DISABLED)
        
        score = int(self.game_max_score)
        if score >= 80:
            self.game_status_var.set(f"KẾT QUẢ: XUẤT SẮC! 🎉 ({score}đ)")
            self.game_feedback_var.set("Tuyệt hảo! Bạn biểu cảm vô cùng chính xác.")
        elif score >= 50:
            self.game_status_var.set(f"KẾT QUẢ: TỐT! 👍 ({score}đ)")
            self.game_feedback_var.set("Khá lắm! Cố gắng tươi/rõ hơn chút nữa nhé.")
        else:
            self.game_status_var.set(f"KẾT QUẢ: THỬ LẠI! 💪 ({score}đ)")
            self.game_feedback_var.set("Hãy thể hiện rõ nét mặt và thử lại!")

    def _stop_game(self) -> None:
        """Force stop the current game session."""

        self.game_active = False
        self.game_stage = "idle"
        self.game_start_btn.configure(state=tk.NORMAL)
        self.game_stop_btn.configure(state=tk.DISABLED)
        
        self.game_status_var.set("Đã dừng trò chơi.")
        self.game_feedback_var.set("Nhấn 'Bắt đầu chơi' để bắt đầu.")
        self.game_score_bar["value"] = 0
        self.game_score_text_var.set("0%")
        self.game_max_score_text_var.set("Cao nhất: 0%")

    def _reset_game_state_ui(self) -> None:
        """Reset game mode labels to initial state."""

        self.game_target_var.set("--")
        self.game_status_var.set("Đang chờ bắt đầu...")
        self.game_feedback_var.set("Nhấn 'Bắt đầu chơi' để bắt đầu.")
        self.game_score_bar["value"] = 0
        self.game_score_text_var.set("0%")
        self.game_max_score_text_var.set("Cao nhất: 0%")

    def _switch_mode(self, mode: str) -> None:
        """Switch sidebar content between Detection mode and Game mode."""

        if self.current_mode == mode:
            return
            
        self.current_mode = mode
        if mode == "detection":
            self._stop_game()
            self.mode_detection_btn.configure(style="Primary.TButton")
            self.mode_game_btn.configure(style="Tool.TButton")
            self.game_view_frame.pack_forget()
            self.detection_view_frame.pack(fill=tk.BOTH, expand=True)
            self.status_var.set("Ready")
        else:
            self.mode_detection_btn.configure(style="Tool.TButton")
            self.mode_game_btn.configure(style="Primary.TButton")
            self.detection_view_frame.pack_forget()
            self.game_view_frame.pack(fill=tk.BOTH, expand=True)
            self.status_var.set("Chế độ chơi game")
            self._reset_game_state_ui()

    def _draw_game_overlay(self, frame: np.ndarray) -> None:
        """Render arcade HUD directly onto the camera BGR image."""
        h, w = frame.shape[:2]
        
        # Top banner background
        banner_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 10, 5), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Translate to ASCII Vietnamese for cv2.putText
        target = self.game_target_emotion
        target_map = {
            "Happy": "HANH PHUC",
            "Sad": "BUON BA",
            "Surprise": "NGAC NHIEN",
            "Anger": "TUC GIAN"
        }
        target_vi = target_map.get(target, "KO XAC DINH")
        
        if self.game_stage == "countdown":
            # Darkened full-screen overlay for countdown
            overlay_full = frame.copy()
            cv2.rectangle(overlay_full, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay_full, 0.70, frame, 0.30, 0, frame)
            
            text_ready = "CHUAN BI..."
            (tw, th), _ = cv2.getTextSize(text_ready, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(
                frame, text_ready,
                ((w - tw) // 2, h // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, cv2.LINE_AA
            )
            
            text_target = f"THACH THUC: {target_vi}"
            (tw2, th2), _ = cv2.getTextSize(text_target, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(
                frame, text_target,
                ((w - tw2) // 2, h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )
            
            # Show large countdown number
            elapsed = time.time() - self.game_start_time
            val = max(1, int(3.0 - elapsed) + 1)
            text_num = str(val)
            (tw3, th3), _ = cv2.getTextSize(text_num, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)
            cv2.putText(
                frame, text_num,
                ((w - tw3) // 2, h // 2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5, cv2.LINE_AA
            )
            
        elif self.game_stage == "playing":
            elapsed = time.time() - self.game_start_time
            remaining = max(0.0, self.game_duration - elapsed)
            
            # Draw Target text on left
            cv2.putText(
                frame, f"MUC TIEU: {target_vi}",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA
            )
            
            # Draw Timer on right
            time_text = f"CON LAI: {remaining:.1f}S"
            (tw, th), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            timer_color = (0, 0, 255) if remaining < 2.0 else (0, 255, 0)
            cv2.putText(
                frame, time_text,
                (w - tw - 20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, timer_color, 2, cv2.LINE_AA
            )
            
            # Draw current score overlay at the bottom
            score_text = f"DIEM: {int(self.game_current_score)}%"
            (tw2, th2), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            
            # Background panel for bottom score
            panel_padding = 12
            cv2.rectangle(
                frame,
                ((w - tw2) // 2 - panel_padding, h - 55),
                ((w - tw2) // 2 + tw2 + panel_padding, h - 15),
                (10, 10, 10),
                -1
            )
            cv2.rectangle(
                frame,
                ((w - tw2) // 2 - panel_padding, h - 55),
                ((w - tw2) // 2 + tw2 + panel_padding, h - 15),
                (0, 215, 255),
                1
            )
            cv2.putText(
                frame, score_text,
                ((w - tw2) // 2, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if self.game_current_score >= 60 else (255, 255, 255), 2, cv2.LINE_AA
            )


def main() -> None:
    root = tk.Tk()
    app = FaceDetectorGui(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        if app.stop_event is not None:
            app.stop_event.set()
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
