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
import customtkinter as ctk
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
    """CustomTkinter UI that keeps model inference off the main UI thread."""

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
        self.root.configure(fg_color="#0b1120")

        self.shell = ctk.CTkFrame(self.root, fg_color="transparent")
        self.shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        self.header = ctk.CTkFrame(self.shell, fg_color="transparent")
        self.header.pack(fill=tk.X, pady=(0, 12))
        ctk.CTkLabel(
            self.header,
            text="Face Analysis",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color="#f8fafc",
        ).pack(side=tk.LEFT)
        self.status_badge = ctk.CTkLabel(
            self.header,
            textvariable=self.status_var,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#dbeafe",
            fg_color="#1d4ed8",
            corner_radius=14,
            padx=14,
            pady=8,
        )
        self.status_badge.pack(side=tk.RIGHT)

        self.workspace = ctk.CTkFrame(self.shell, fg_color="transparent")
        self.workspace.pack(fill=tk.BOTH, expand=True)

        self.main_panel = ctk.CTkFrame(self.workspace, fg_color="transparent")
        self.main_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.side_panel = ctk.CTkFrame(
            self.workspace,
            width=390,
            fg_color="#111827",
            corner_radius=20,
            border_width=1,
            border_color="#243041",
        )
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))
        self.side_panel.pack_propagate(False)

        preview_shell = ctk.CTkFrame(
            self.main_panel,
            fg_color="#111827",
            corner_radius=22,
            border_width=1,
            border_color="#243041",
        )
        preview_shell.pack(fill=tk.BOTH, expand=True)
        preview_shell.pack_propagate(False)

        self.preview_header = ctk.CTkFrame(preview_shell, fg_color="transparent")
        self.preview_header.pack(fill=tk.X, padx=16, pady=(14, 8))
        ctk.CTkLabel(
            self.preview_header,
            text="Live Preview",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            text_color="#f8fafc",
        ).pack(side=tk.LEFT)
        self.preview_state_chip = ctk.CTkLabel(
            self.preview_header,
            textvariable=self.result_state_var,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#bfdbfe",
            fg_color="#1e3a8a",
            corner_radius=12,
            padx=12,
            pady=6,
        )
        self.preview_state_chip.pack(side=tk.RIGHT)

        preview_frame = tk.Frame(
            preview_shell,
            bg="#020617",
            highlightbackground="#263449",
            highlightthickness=1,
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
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

        self.metrics = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self.metrics.pack(fill=tk.X, pady=(12, 0))
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
        parent.grid_columnconfigure(column, weight=1, uniform="metrics")
        cell = ctk.CTkFrame(
            parent,
            fg_color="#111827",
            corner_radius=18,
            border_width=1,
            border_color="#243041",
        )
        cell.grid(row=0, column=column, sticky=tk.EW, padx=(0 if column == 0 else 10, 0))
        ctk.CTkLabel(
            cell,
            text=title,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#94a3b8",
        ).pack(anchor=tk.W, padx=14, pady=(12, 2))
        ctk.CTkLabel(
            cell,
            textvariable=variable,
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#f8fafc",
        ).pack(anchor=tk.W, padx=14, pady=(0, 12))

    def _build_side_panel_content(self, parent) -> None:
        """Build side panel with Mode Switcher and container frames."""
        self.mode_switcher = ctk.CTkFrame(parent, fg_color="transparent")
        self.mode_switcher.pack(fill=tk.X, padx=14, pady=(14, 12))
        self.mode_switcher.grid_columnconfigure((0, 1), weight=1)

        self.mode_detection_btn = ctk.CTkButton(
            self.mode_switcher,
            text="Nhận diện",
            command=lambda: self._switch_mode("detection"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            text_color="#ffffff",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            corner_radius=14,
            height=42,
        )
        self.mode_detection_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 6))

        self.mode_game_btn = ctk.CTkButton(
            self.mode_switcher,
            text="Trò chơi",
            command=lambda: self._switch_mode("game"),
            fg_color="#334155",
            hover_color="#475569",
            text_color="#f8fafc",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            corner_radius=14,
            height=42,
        )
        self.mode_game_btn.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))

        self.side_content_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.side_content_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        self.detection_view_frame = ctk.CTkScrollableFrame(
            self.side_content_frame,
            fg_color="transparent",
            scrollbar_button_color="#334155",
            scrollbar_button_hover_color="#475569",
        )
        self._build_controls(self.detection_view_frame)

        self._build_results(self.detection_view_frame)

        self.history_container = ctk.CTkFrame(self.detection_view_frame, fg_color="transparent")
        self.history_container.pack(fill=tk.X, pady=(4, 0))
        self._build_history(self.history_container)

        self.detection_view_frame.pack(fill=tk.BOTH, expand=True)

        self.game_view_frame = ctk.CTkFrame(self.side_content_frame, fg_color="transparent")
        self._build_game_ui(self.game_view_frame)

    def _build_controls(self, parent) -> None:
        ctk.CTkLabel(
            parent,
            text="Controls",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#f8fafc",
        ).pack(anchor=tk.W, pady=(0, 8))

        controls_card = ctk.CTkFrame(
            parent,
            fg_color="#0f172a",
            corner_radius=18,
            border_width=1,
            border_color="#243041",
        )
        controls_card.pack(fill=tk.X)

        camera_row = ctk.CTkFrame(controls_card, fg_color="transparent")
        camera_row.pack(fill=tk.X, padx=14, pady=(14, 10))
        ctk.CTkLabel(
            camera_row,
            text="Camera",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cbd5e1",
        ).pack(side=tk.LEFT)
        self.camera_entry = ctk.CTkEntry(
            camera_row,
            width=72,
            textvariable=self.camera_index_var,
            justify=tk.CENTER,
            corner_radius=12,
        )
        self.camera_entry.pack(side=tk.RIGHT)

        self.detector_combo = self._control_combo(
            controls_card,
            "Detector",
            self.detector_backend_var,
            ("YOLO", "Auto", "Haar"),
        )

        self.webcam_button = ctk.CTkButton(
            controls_card,
            text="Start Camera",
            command=self.start_webcam,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            text_color="#ffffff",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            corner_radius=14,
            height=42,
        )
        self.webcam_button.pack(fill=tk.X, padx=14, pady=(14, 10))

        action_grid = ctk.CTkFrame(controls_card, fg_color="transparent")
        action_grid.pack(fill=tk.X, padx=14, pady=(0, 14))
        action_grid.grid_columnconfigure((0, 1), weight=1)
        self.snapshot_button = ctk.CTkButton(
            action_grid,
            text="Snapshot",
            command=self.snapshot_camera,
            fg_color="#334155",
            hover_color="#475569",
            text_color="#f8fafc",
            corner_radius=12,
            height=38,
        )
        self.snapshot_button.grid(row=0, column=0, sticky=tk.EW, padx=(0, 6), pady=(0, 8))
        self.stop_button = ctk.CTkButton(
            action_grid,
            text="Stop",
            command=self.stop_current,
            fg_color="#dc2626",
            hover_color="#b91c1c",
            text_color="#ffffff",
            corner_radius=12,
            height=38,
        )
        self.stop_button.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0), pady=(0, 8))

        self.image_button = ctk.CTkButton(
            action_grid,
            text="Open Image",
            command=self.choose_image,
            fg_color="#334155",
            hover_color="#475569",
            text_color="#f8fafc",
            corner_radius=12,
            height=38,
        )
        self.image_button.grid(row=1, column=0, sticky=tk.EW, padx=(0, 6))
        self.save_button = ctk.CTkButton(
            action_grid,
            text="Save Result",
            command=self.save_current_result,
            fg_color="#334155",
            hover_color="#475569",
            text_color="#f8fafc",
            corner_radius=12,
            height=38,
        )
        self.save_button.grid(row=1, column=1, sticky=tk.EW, padx=(6, 0))

        self.quit_button = ctk.CTkButton(
            parent,
            text="Exit",
            command=self.close,
            fg_color="#1e293b",
            hover_color="#334155",
            text_color="#f8fafc",
            corner_radius=14,
            height=40,
        )
        self.quit_button.pack(fill=tk.X, pady=(12, 0))

    def _control_combo(
        self,
        parent,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ):
        ctk.CTkLabel(
            parent,
            text=label,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#cbd5e1",
        ).pack(anchor=tk.W, padx=14, pady=(0, 6))
        combo = ctk.CTkComboBox(
            parent,
            variable=variable,
            values=list(values),
            state="readonly",
            justify=tk.CENTER,
            corner_radius=12,
            button_color="#2563eb",
            button_hover_color="#1d4ed8",
            border_color="#334155",
            fg_color="#0f172a",
            dropdown_fg_color="#0f172a",
            dropdown_hover_color="#1e293b",
        )
        combo.pack(fill=tk.X, padx=14)
        combo.configure(command=lambda _value: self._sync_active_detector())
        return combo

    def _build_results(self, parent) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill=tk.X, pady=(18, 8))
        ctk.CTkLabel(
            header,
            text="Results",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#f8fafc",
        ).pack(side=tk.LEFT)
        ctk.CTkLabel(
            header,
            textvariable=self.result_state_var,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#94a3b8",
        ).pack(side=tk.RIGHT)

        self.results_scroll = ctk.CTkFrame(parent, fg_color="#0f172a", corner_radius=14)
        self.results_scroll.pack(fill=tk.X)
        self.results_scroll.content = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        self.results_scroll.content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._render_predictions([])

    def _build_history(self, parent) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill=tk.X, pady=(18, 8))
        ctk.CTkLabel(
            header,
            text="History",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#f8fafc",
        ).pack(side=tk.LEFT)
        self.clear_history_button = ctk.CTkButton(
            header,
            text="Clear",
            command=self.clear_history,
            fg_color="#334155",
            hover_color="#475569",
            text_color="#f8fafc",
            corner_radius=10,
            height=30,
            width=70,
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
        style.configure("Title.TLabel", background="#0b1120", foreground="#f8fafc", font=("Segoe UI", 20, "bold"))
        style.configure("Status.TLabel", background="#0b1120", foreground="#60a5fa", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#111827", foreground="#f8fafc", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background="#111827", foreground="#cbd5e1", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background="#111827", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background="#172033", foreground="#f8fafc", font=("Segoe UI", 10, "bold"))
        style.configure("CardBody.TLabel", background="#172033", foreground="#cbd5e1", font=("Segoe UI", 9))
        style.configure("CardMuted.TLabel", background="#172033", foreground="#94a3b8", font=("Segoe UI", 8))
        style.configure("MetricTitle.TLabel", background="#111827", foreground="#94a3b8", font=("Segoe UI", 8, "bold"))
        style.configure("MetricValue.TLabel", background="#111827", foreground="#f8fafc", font=("Segoe UI", 16, "bold"))
        style.configure("Primary.TButton", background="#3b82f6", foreground="#ffffff")
        style.configure("Tool.TButton", background="#334155", foreground="#f8fafc")
        style.configure("Danger.TButton", background="#ef4444", foreground="#ffffff")
        style.configure("Small.TButton", background="#334155", foreground="#f8fafc")
        style.configure("Result.Horizontal.TProgressbar", troughcolor="#334155", background="#10b981")
        style.configure(
            "TScrollbar",
            troughcolor="#111827",
            background="#334155",
            arrowcolor="#f8fafc",
            bordercolor="#111827",
            darkcolor="#111827",
            lightcolor="#111827",
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
            self._set_widget_state(widget, state)
        self._set_widget_state(self.detector_combo, "readonly" if enabled else tk.DISABLED)
        self._set_widget_state(self.snapshot_button, tk.DISABLED)
        self._set_widget_state(self.stop_button, tk.DISABLED)
        self._set_save_enabled(enabled and self.latest_result is not None)
        self._set_widget_state(self.quit_button, tk.NORMAL)

    def _set_camera_running(self) -> None:
        for widget in (
            self.camera_entry,
            self.webcam_button,
            self.detector_combo,
        ):
            widget.configure(state=tk.DISABLED)
        self._set_widget_state(self.image_button, tk.NORMAL)
        self.snapshot_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL)
        self._set_widget_state(self.save_button, tk.DISABLED)
        self._set_widget_state(self.quit_button, tk.NORMAL)

    def _set_processing_running(self) -> None:
        for widget in (
            self.camera_entry,
            self.webcam_button,
            self.image_button,
            self.detector_combo,
            self.snapshot_button,
        ):
            widget.configure(state=tk.DISABLED)
        self._set_widget_state(self.stop_button, tk.DISABLED)
        self._set_widget_state(self.save_button, tk.DISABLED)
        self._set_widget_state(self.quit_button, tk.NORMAL)

    def _set_idle_controls(self) -> None:
        if self.models is None:
            self._set_controls_enabled(False)
            return
        self._set_widget_state(self.camera_entry, tk.NORMAL)
        self._set_widget_state(self.webcam_button, tk.NORMAL)
        self._set_widget_state(self.image_button, tk.NORMAL)
        self._set_widget_state(self.detector_combo, "readonly")
        self._set_widget_state(self.snapshot_button, tk.DISABLED)
        self._set_widget_state(self.stop_button, tk.DISABLED)
        self.stop_button.configure(text="Stop")
        self._set_save_enabled(self.latest_result is not None)
        self._set_widget_state(self.quit_button, tk.NORMAL)

    def _set_save_enabled(self, enabled: bool) -> None:
        self._set_widget_state(self.save_button, tk.NORMAL if enabled else tk.DISABLED)

    def _set_widget_state(self, widget, state) -> None:
        normalized = state
        if state == tk.NORMAL:
            normalized = "normal"
        elif state == tk.DISABLED:
            normalized = "disabled"
        try:
            widget.configure(state=normalized)
        except tk.TclError:
            widget.configure(state=state)

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

                if now - last_ui_update >= 0.35:
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
            self.game_score_bar.set(max(0.0, min(1.0, score / 100.0)))
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
            self._set_widget_state(self.stop_button, tk.DISABLED)
            self._set_widget_state(self.snapshot_button, tk.DISABLED)

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

    def _emotion_badge_colors(self, emotion: str) -> tuple[str, str]:
        key = (emotion or '').strip().lower()
        mapping = {
            'happy': ('#064e3b', '#d1fae5'),
            'sad': ('#1e3a8a', '#dbeafe'),
            'surprise': ('#78350f', '#fef3c7'),
            'anger': ('#7f1d1d', '#fee2e2'),
            'angry': ('#7f1d1d', '#fee2e2'),
            'neutral': ('#374151', '#e5e7eb'),
            'fear': ('#581c87', '#f3e8ff'),
            'disgust': ('#14532d', '#dcfce7'),
        }
        return mapping.get(key, ('#0f766e', '#ccfbf1'))

    def _confidence_bar_color(self, label: str) -> str:
        return {
            'Emotion': '#22c55e',
            'Gender': '#38bdf8',
            'Age': '#a78bfa',
        }.get(label, '#22c55e')

    def _render_predictions(self, predictions: list[FacePrediction]) -> None:
        if self.current_mode == "game":
            return

        current_count = len(predictions)
        cached_count = len(self.face_cards)

        if current_count != cached_count:
            for child in self.results_scroll.content.winfo_children():
                child.destroy()
            self.result_photo_refs.clear()
            self.face_cards.clear()

            if not predictions:
                empty = ctk.CTkLabel(
                    self.results_scroll.content,
                    text="No face details",
                    font=ctk.CTkFont(family="Segoe UI", size=12),
                    text_color="#94a3b8",
                )
                empty.pack(fill=tk.X, padx=12, pady=14)
                return

            for index, prediction in enumerate(predictions, start=1):
                card_refs = self._add_prediction_card(index, prediction)
                self.face_cards.append(card_refs)
            return

        for card_refs, prediction in zip(self.face_cards, predictions):
            result = prediction.result
            self._update_prediction_card(card_refs, prediction, result)

    def _add_prediction_card(self, index: int, prediction: FacePrediction) -> dict:
        result = prediction.result
        card = ctk.CTkFrame(
            self.results_scroll.content,
            fg_color="#162033",
            corner_radius=18,
            border_width=1,
            border_color="#263449",
        )
        card.pack(fill=tk.X, pady=(0, 10), padx=2, ipadx=0, ipady=0)
        card.grid_columnconfigure(1, weight=1)

        media = ctk.CTkFrame(card, fg_color="transparent")
        media.grid(row=0, column=0, sticky=tk.NW, padx=(12, 10), pady=10)
        if prediction.crop_bgr is not None:
            photo = self._make_crop_photo(prediction.crop_bgr)
            self.result_photo_refs.append(photo)
            crop_label = tk.Label(media, image=photo, bg="#182235", bd=0, highlightthickness=0)
            crop_label.image = photo
        else:
            crop_label = tk.Label(
                media,
                text=f"Face {index}",
                width=11,
                height=6,
                bg="#0f172a",
                fg="#94a3b8",
                font=("Segoe UI", 10, "bold"),
                bd=0,
                highlightthickness=0,
            )
        crop_label.pack()

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.grid(row=0, column=1, sticky=tk.NSEW, padx=(0, 12), pady=10)
        body.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(body, fg_color="transparent")
        header.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        header.grid_columnconfigure(0, weight=1)
        title_label = ctk.CTkLabel(
            header,
            text=f"Face {index}",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color="#f8fafc",
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        badge_bg, badge_fg = self._emotion_badge_colors(result.emotion)
        emotion_badge = ctk.CTkLabel(
            header,
            text=result.emotion,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=badge_fg,
            fg_color=badge_bg,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        emotion_badge.grid(row=0, column=1, sticky=tk.E)

        e_refs = self._add_confidence_row(body, 1, "Emotion", result.emotion_confidence, result.emotion)
        g_refs = self._add_confidence_row(body, 2, "Gender", result.gender_confidence, result.gender)
        a_refs = self._add_confidence_row(body, 3, "Age", result.age_confidence, result.age)

        x, y, w, h = prediction.box
        footer = ctk.CTkFrame(body, fg_color="transparent")
        footer.grid(row=4, column=0, sticky=tk.EW, pady=(8, 0))
        box_label = ctk.CTkLabel(
            footer,
            text=f"Box {w}x{h} at {x}, {y}",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#94a3b8",
        )
        box_label.pack(anchor=tk.W)

        return {
            "crop_label": crop_label,
            "title_label": title_label,
            "emotion_badge": emotion_badge,
            "box_label": box_label,
            "emotion": e_refs,
            "gender": g_refs,
            "age": a_refs,
            "index": index,
        }

    def _update_prediction_card(self, card_refs: dict, prediction: FacePrediction, result) -> None:
        index = card_refs["index"]
        card_refs["title_label"].configure(text=f"Face {index}")
        badge_bg, badge_fg = self._emotion_badge_colors(result.emotion)
        card_refs["emotion_badge"].configure(text=result.emotion, fg_color=badge_bg, text_color=badge_fg)

        if prediction.crop_bgr is not None:
            photo = self._make_crop_photo(prediction.crop_bgr)
            self.result_photo_refs.append(photo)
            card_refs["crop_label"].configure(image=photo)
            card_refs["crop_label"].image = photo

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
        percent = max(0, min(100, int(round(float(confidence) * 100))))
        wrapper = ctk.CTkFrame(parent, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky=tk.EW, pady=(0, 6))
        wrapper.grid_columnconfigure(1, weight=1)

        top = ctk.CTkFrame(wrapper, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=3, sticky=tk.EW)
        top.grid_columnconfigure(1, weight=1)
        label_widget = ctk.CTkLabel(
            top,
            text=label,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#cbd5e1",
            width=58,
        )
        label_widget.grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        value_label = ctk.CTkLabel(
            top,
            text=value,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#93c5fd" if label == "Age" else "#cbd5e1",
        )
        value_label.grid(row=0, column=1, sticky=tk.W)
        pct_label = ctk.CTkLabel(
            top,
            text=f"{percent}%",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#f8fafc",
            width=42,
        )
        pct_label.grid(row=0, column=2, sticky=tk.E)

        bar = ctk.CTkProgressBar(wrapper, progress_color=self._confidence_bar_color(label), fg_color="#334155", corner_radius=999, height=10)
        bar.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(3, 0))
        bar.set(percent / 100.0)
        return {"bar": bar, "pct_label": pct_label, "val_label": value_label}

    def _update_confidence_row(self, refs: dict, confidence: float, value: str) -> None:
        percent = max(0, min(100, int(round(float(confidence) * 100))))
        refs["bar"].set(percent / 100.0)
        refs["pct_label"].configure(text=f"{percent}%")
        refs["val_label"].configure(text=value)

    def _make_crop_photo(self, crop_bgr: np.ndarray) -> ImageTk.PhotoImage:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        image = ImageOps.fit(image, (92, 92), method=Image.Resampling.BILINEAR)
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
            yolo_max_det=20,
        )

    def _build_game_ui(self, parent) -> None:
        """Create UI elements for the Emotion Mimic Game."""
        ctk.CTkLabel(
            parent,
            text="Trò chơi bắt chước",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#f8fafc",
        ).pack(anchor=tk.W, pady=(0, 10))

        instr_card = ctk.CTkFrame(parent, fg_color="#172033", corner_radius=16, border_width=1, border_color="#243041")
        instr_card.pack(fill=tk.X, pady=(0, 14))
        ctk.CTkLabel(
            instr_card,
            text="Hãy bắt chước cảm xúc trước camera",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#cbd5e1",
            wraplength=320,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=12, pady=12)

        target_card = ctk.CTkFrame(parent, fg_color="#172033", corner_radius=16, border_width=1, border_color="#243041")
        target_card.pack(fill=tk.X, pady=(0, 14))
        ctk.CTkLabel(target_card, text="Mục tiêu của biểu cảm", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#94a3b8").pack(anchor=tk.CENTER, pady=(12, 4))
        self.game_target_label = ctk.CTkLabel(
            target_card,
            textvariable=self.game_target_var,
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color="#f59e0b",
        )
        self.game_target_label.pack(anchor=tk.CENTER, pady=(0, 12))

        score_card = ctk.CTkFrame(parent, fg_color="#172033", corner_radius=16, border_width=1, border_color="#243041")
        score_card.pack(fill=tk.X, pady=(0, 14))
        score_row = ctk.CTkFrame(score_card, fg_color="transparent")
        score_row.pack(fill=tk.X, padx=14, pady=(12, 6))
        ctk.CTkLabel(score_row, text="Điểm", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#cbd5e1").pack(side=tk.LEFT)
        ctk.CTkLabel(score_row, textvariable=self.game_score_text_var, font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color="#f8fafc").pack(side=tk.RIGHT)
        self.game_score_bar = ctk.CTkProgressBar(score_card, progress_color="#10b981", fg_color="#334155", corner_radius=999, height=12)
        self.game_score_bar.pack(fill=tk.X, padx=14, pady=(0, 8))
        self.game_score_bar.set(0)
        ctk.CTkLabel(score_card, textvariable=self.game_max_score_text_var, font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#94a3b8").pack(anchor=tk.W, padx=14, pady=(0, 12))

        status_card = ctk.CTkFrame(parent, fg_color="#172033", corner_radius=16, border_width=1, border_color="#243041")
        status_card.pack(fill=tk.X, pady=(0, 18))
        ctk.CTkLabel(status_card, text="Trạng thái", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#94a3b8").pack(anchor=tk.W, padx=12, pady=(12, 2))
        ctk.CTkLabel(status_card, textvariable=self.game_status_var, font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color="#f8fafc").pack(anchor=tk.W, padx=12, pady=(0, 4))
        ctk.CTkLabel(status_card, textvariable=self.game_feedback_var, font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#cbd5e1", wraplength=320, justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=(0, 12))

        self.game_start_btn = ctk.CTkButton(parent, text="Bắt đầu chơi", command=self._start_game, fg_color="#2563eb", hover_color="#1d4ed8", text_color="#ffffff", corner_radius=14, height=42)
        self.game_start_btn.pack(fill=tk.X, pady=(0, 8))
        self.game_stop_btn = ctk.CTkButton(parent, text="Dừng chơi", command=self._stop_game, fg_color="#dc2626", hover_color="#b91c1c", text_color="#ffffff", corner_radius=14, height=42)
        self.game_stop_btn.pack(fill=tk.X)
        self._set_widget_state(self.game_stop_btn, tk.DISABLED)

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
        
        self._set_widget_state(self.game_start_btn, tk.DISABLED)
        self._set_widget_state(self.game_stop_btn, tk.NORMAL)
        
        self.game_status_var.set("Chuẩn bị...")
        self.game_feedback_var.set("Chuẩn bị biểu diễn nét mặt!")
        self.game_score_bar.set(0)
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
        self._set_widget_state(self.game_start_btn, tk.NORMAL)
        self._set_widget_state(self.game_stop_btn, tk.DISABLED)
        
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
        self._set_widget_state(self.game_start_btn, tk.NORMAL)
        self._set_widget_state(self.game_stop_btn, tk.DISABLED)
        
        self.game_status_var.set("Đã dừng trò chơi.")
        self.game_feedback_var.set("Nhấn 'Bắt đầu chơi' để bắt đầu.")
        self.game_score_bar.set(0)
        self.game_score_text_var.set("0%")
        self.game_max_score_text_var.set("Cao nhất: 0%")

    def _reset_game_state_ui(self) -> None:
        """Reset game mode labels to initial state."""
        self.game_target_var.set("--")
        self.game_status_var.set("Đang chờ bắt đầu...")
        self.game_feedback_var.set("Nhấn 'Bắt đầu chơi' để bắt đầu.")
        self.game_score_bar.set(0)
        self.game_score_text_var.set("0%")
        self.game_max_score_text_var.set("Cao nhất: 0%")

    def _switch_mode(self, mode: str) -> None:
        """Switch sidebar content between Detection mode and Game mode."""
        if self.current_mode == mode:
            return
            
        self.current_mode = mode
        if mode == "detection":
            self._stop_game()
            self.mode_detection_btn.configure(fg_color="#2563eb", hover_color="#1d4ed8")
            self.mode_game_btn.configure(fg_color="#334155", hover_color="#475569")
            self.game_view_frame.pack_forget()
            self.detection_view_frame.pack(fill=tk.BOTH, expand=True)
            self.status_var.set("Ready")
        else:
            self.mode_detection_btn.configure(fg_color="#334155", hover_color="#475569")
            self.mode_game_btn.configure(fg_color="#2563eb", hover_color="#1d4ed8")
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
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
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
