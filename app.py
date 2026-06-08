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
            bg="#111827",
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
        models.gender_model.predict(dummy, verbose=0)
        models.age_model.predict(dummy, verbose=0)
        models.emotion_model.predict(dummy, verbose=0)
    except Exception as exc:
        logger.warning("Could not warm up models: %s", exc)


def predict_face(face_bgr: np.ndarray, models: Models) -> FaceResult:
    """Run all predictions on one face crop."""
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


def predict_emotion_only(face_bgr: np.ndarray, models: Models) -> tuple[str, float]:
    """Predict only emotion for faster realtime updates."""
    emotion = predict_emotion(
        face_bgr,
        models.emotion_model,
        image_size=IMAGE_SIZE,
    )
    return emotion["emotion"], emotion["emotion_confidence"]


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

        self._build_ui()
        self._set_controls_enabled(False)
        self._start_model_loader()
        self._pump_frames()

    def _build_ui(self) -> None:
        self._configure_styles()
        self.root.configure(bg="#0b1120")

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
        """Build side panel with Controls at top, History at bottom, Results in middle."""
        # Controls at the top
        self._build_controls(parent)

        # History at the bottom — pack BEFORE results so it claims bottom space
        self.history_container = ttk.Frame(parent, style="Panel.TFrame")
        self.history_container.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_history(self.history_container)

        # Results fills the remaining middle space
        self._build_results(parent)

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

        style.configure("App.TFrame", background="#0b1120")
        style.configure("Panel.TFrame", background="#111827")
        style.configure("Metric.TFrame", background="#111827")
        style.configure("Card.TFrame", background="#172033")
        style.configure(
            "Title.TLabel",
            background="#0b1120",
            foreground="#f8fafc",
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background="#0b1120",
            foreground="#93c5fd",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background="#111827",
            foreground="#f8fafc",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#111827",
            foreground="#cbd5e1",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background="#111827",
            foreground="#94a3b8",
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardTitle.TLabel",
            background="#172033",
            foreground="#f8fafc",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "CardBody.TLabel",
            background="#172033",
            foreground="#cbd5e1",
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardMuted.TLabel",
            background="#172033",
            foreground="#94a3b8",
            font=("Segoe UI", 8),
        )
        style.configure(
            "MetricTitle.TLabel",
            background="#111827",
            foreground="#94a3b8",
            font=("Segoe UI", 8, "bold"),
        )
        style.configure(
            "MetricValue.TLabel",
            background="#111827",
            foreground="#f8fafc",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Primary.TButton",
            background="#2563eb",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10, "bold"),
            padding=(16, 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#475569")],
            foreground=[("disabled", "#cbd5e1")],
        )
        style.configure(
            "Tool.TButton",
            background="#1e293b",
            foreground="#f8fafc",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10),
            padding=(14, 9),
        )
        style.map(
            "Tool.TButton",
            background=[("active", "#334155"), ("disabled", "#334155")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure(
            "Danger.TButton",
            background="#be123c",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10, "bold"),
            padding=(14, 9),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#9f1239"), ("disabled", "#475569")],
            foreground=[("disabled", "#cbd5e1")],
        )
        style.configure(
            "Small.TButton",
            background="#1e293b",
            foreground="#f8fafc",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 8),
            padding=(8, 4),
        )
        style.configure("Result.Horizontal.TProgressbar", troughcolor="#334155", background="#22c55e")

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
            full_inference_interval=0.5,
            detection_width=640,
            max_faces=3,
            emotion_history_size=3,
            emotion_min_confidence=0.35,
            label_switch_margin=0.15,
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
        for child in self.results_scroll.content.winfo_children():
            child.destroy()
        self.result_photo_refs.clear()

        if not predictions:
            empty = ttk.Label(
                self.results_scroll.content,
                text="No face details",
                style="Muted.TLabel",
                padding=(12, 14),
            )
            empty.pack(fill=tk.X)
            return

        for index, prediction in enumerate(predictions, start=1):
            self._add_prediction_card(index, prediction)

    def _add_prediction_card(self, index: int, prediction: FacePrediction) -> None:
        result = prediction.result
        card = ttk.Frame(self.results_scroll.content, style="Card.TFrame", padding=(10, 10, 10, 10))
        card.pack(fill=tk.X, pady=(0, 10))
        card.columnconfigure(1, weight=1)

        if prediction.crop_bgr is not None:
            photo = self._make_crop_photo(prediction.crop_bgr)
            self.result_photo_refs.append(photo)
            crop_label = ttk.Label(card, image=photo, style="CardBody.TLabel")
        else:
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

        ttk.Label(
            card,
            text=f"Face {index} - {result.emotion}",
            style="CardTitle.TLabel",
        ).grid(row=0, column=1, sticky=tk.EW)
        self._add_confidence_row(card, 1, "Emotion", result.emotion_confidence, result.emotion)
        self._add_confidence_row(card, 2, "Gender", result.gender_confidence, result.gender)
        self._add_confidence_row(card, 3, "Age", result.age_confidence, result.age)

        x, y, w, h = prediction.box
        ttk.Label(
            card,
            text=f"Box {w}x{h} at {x}, {y}",
            style="CardMuted.TLabel",
        ).grid(row=4, column=1, sticky=tk.EW, pady=(4, 0))

    def _add_confidence_row(
        self,
        parent,
        row: int,
        label: str,
        confidence: float,
        value: str,
    ) -> None:
        percent = max(0, min(100, int(round(float(confidence) * 100))))
        wrapper = ttk.Frame(parent, style="Card.TFrame")
        wrapper.grid(row=row, column=1, sticky=tk.EW, pady=(6, 0))
        wrapper.columnconfigure(1, weight=1)
        ttk.Label(wrapper, text=label, style="CardBody.TLabel", width=8).grid(
            row=0,
            column=0,
            sticky=tk.W,
        )
        ttk.Progressbar(
            wrapper,
            maximum=100,
            value=percent,
            style="Result.Horizontal.TProgressbar",
        ).grid(row=0, column=1, sticky=tk.EW, padx=(8, 8))
        ttk.Label(wrapper, text=f"{percent}%", style="CardBody.TLabel", width=5).grid(
            row=0,
            column=2,
            sticky=tk.E,
        )
        ttk.Label(wrapper, text=value, style="CardMuted.TLabel").grid(
            row=1,
            column=0,
            columnspan=3,
            sticky=tk.W,
            pady=(2, 0),
        )

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

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = ImageOps.contain(
            image,
            self.preview_size,
            method=Image.Resampling.BILINEAR,
        )

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
            yolo_imgsz=640,
            yolo_confidence=0.28,
            yolo_iou=0.45,
            yolo_max_det=20,
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
