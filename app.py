"""Button-based desktop UI for face age, gender, and emotion detection."""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps

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


def process_image_frame(frame: np.ndarray, models: Models, config: FaceDetectionConfig) -> int:
    """Detect and annotate all faces in a still image."""
    boxes = detect_faces_bgr(frame, config)
    processed = 0
    for box in boxes:
        try:
            face = crop_face_bgr(frame, box, margin=config.margin)
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
        processed += 1
    return processed


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
        self.root.geometry("1120x820")
        self.root.minsize(920, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._maximize_window)

        self.models: Optional[object] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.frame_queue: queue.Queue[Image.Image] = queue.Queue(maxsize=1)
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_size = DISPLAY_SIZE
        self.last_display_time = 0.0
        self.camera_view_active = False
        self.pending_image_open = False

        self.status_var = tk.StringVar(value="Loading models...")
        self.camera_index_var = tk.StringVar(value="0")
        self.detector_backend_var = tk.StringVar(value="YOLO")

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

        preview_frame = tk.Frame(
            self.shell,
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
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._exit_fullscreen)
        self.root.bind("<Return>", self._stop_from_key)
        self.root.bind("<space>", self._stop_from_key)

        self.controls = ttk.Frame(self.shell, style="App.TFrame")
        self.controls.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0), before=preview_frame)

        ttk.Label(self.controls, text="Camera Index", style="Body.TLabel").pack(side=tk.LEFT)
        self.camera_entry = ttk.Entry(
            self.controls,
            width=6,
            textvariable=self.camera_index_var,
            justify=tk.CENTER,
        )
        self.camera_entry.pack(side=tk.LEFT, padx=(8, 18), ipady=3)

        ttk.Label(self.controls, text="Detector", style="Body.TLabel").pack(side=tk.LEFT)
        self.detector_combo = ttk.Combobox(
            self.controls,
            width=8,
            textvariable=self.detector_backend_var,
            values=("YOLO", "Auto", "Haar"),
            state="readonly",
            justify=tk.CENTER,
        )
        self.detector_combo.pack(side=tk.LEFT, padx=(8, 18), ipady=3)

        self.webcam_button = ttk.Button(
            self.controls,
            text="Start Camera",
            command=self.start_webcam,
            style="Primary.TButton",
        )
        self.webcam_button.pack(side=tk.LEFT, padx=5)

        self.image_button = ttk.Button(
            self.controls,
            text="Open Image",
            command=self.choose_image,
            style="Tool.TButton",
        )
        self.image_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            self.controls,
            text="Stop",
            command=self.stop_current,
            style="Danger.TButton",
        )
        self.stop_button.pack(side=tk.LEFT, padx=(18, 5))

        self.quit_button = ttk.Button(
            self.controls,
            text="Exit",
            command=self.close,
            style="Tool.TButton",
        )
        self.quit_button.pack(side=tk.RIGHT)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#0b1120")
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
            "Body.TLabel",
            background="#0b1120",
            foreground="#cbd5e1",
            font=("Segoe UI", 10),
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
        self.stop_button.configure(state=tk.DISABLED)

    def _set_running(self, running: bool) -> None:
        main_state = tk.DISABLED if running else tk.NORMAL
        for widget in (
            self.camera_entry,
            self.webcam_button,
            self.image_button,
        ):
            widget.configure(state=main_state)
        self.detector_combo.configure(state=tk.DISABLED if running else "readonly")
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)

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

        config = self._face_detection_config(
            min_neighbors=4,
            min_size=(32, 32),
            margin=0.30,
        )
        self.status_var.set("Camera running")
        self._enter_camera_view()
        self._start_capture_worker(cap, config)

    def choose_image(self) -> None:
        if self.camera_view_active and self.stop_event is not None:
            self.pending_image_open = True
            self.stop_current()
            return

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

        self.status_var.set("Processing image")
        self._set_running(True)
        self.stop_button.configure(state=tk.DISABLED)
        config = self._face_detection_config()
        self.worker_thread = threading.Thread(
            target=self._image_worker,
            args=(Path(file_path), config),
            daemon=True,
        )
        self.worker_thread.start()

    def _start_capture_worker(
        self,
        cap: cv2.VideoCapture,
        config: FaceDetectionConfig,
    ) -> None:
        self.stop_event = threading.Event()
        self._set_running(True)
        self.image_button.configure(state=tk.NORMAL)
        self.quit_button.configure(state=tk.NORMAL)
        self.stop_button.configure(text="Stop Camera")
        self.worker_thread = threading.Thread(
            target=self._capture_worker,
            args=(cap, self.stop_event, config),
            daemon=True,
        )
        self.worker_thread.start()

    def _capture_worker(
        self,
        cap: cv2.VideoCapture,
        stop_event: threading.Event,
        config: FaceDetectionConfig,
    ) -> None:
        processor = AsyncRealtimeFaceProcessor(
            self.models,
            predict_face,
            config,
            predict_emotion=predict_emotion_only,
            partial_result_factory=make_partial_face_result,
            inference_interval=0.18,
            full_inference_interval=1.6,
            detection_width=640,
            max_faces=3,
            emotion_history_size=5,
            emotion_min_confidence=0.30,
            label_switch_margin=0.18,
            label_switch_count=3,
        )
        prev_time = time.time()

        try:
            processor.start()
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                processor.process_frame(frame)
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                draw_fps(frame, fps)

                self._queue_frame(frame)
                time.sleep(0.001)
        finally:
            processor.stop()
            cap.release()

        if stop_event.is_set():
            message = "Stopped"
        else:
            message = "Camera ended"
        self._run_on_ui(self._finish_worker, message)

    def _image_worker(self, path: Path, config: FaceDetectionConfig) -> None:
        frame = cv2.imread(str(path))
        if frame is None:
            self._run_on_ui(self._finish_worker, f"Could not read image: {path}")
            return

        face_count = process_image_frame(frame, self.models, config)
        output_path = path.with_name(f"{path.stem}_result{path.suffix}")
        cv2.imwrite(str(output_path), frame)
        self._queue_frame(frame)

        if face_count == 0:
            message = "No faces detected in the image"
        else:
            message = f"Saved image result: {output_path}"
        self._run_on_ui(self._finish_worker, message)

    def _finish_worker(self, message: str) -> None:
        self.status_var.set(message)
        self.stop_event = None
        self._exit_camera_view()
        if self.models is not None:
            self._set_running(False)
            self.stop_button.configure(text="Stop")
        if self.pending_image_open:
            self.pending_image_open = False
            self.root.after(150, self.choose_image)

    def stop_current(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("Stopping")
            self.stop_button.configure(state=tk.DISABLED)

    def close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self.root.after(100, self.root.destroy)

    def _queue_frame(self, frame_bgr) -> None:
        now = time.time()
        if now - self.last_display_time < DISPLAY_INTERVAL:
            return
        self.last_display_time = now

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        target_size = self.preview_size
        image = ImageOps.contain(
            image,
            target_size,
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

    def _face_detection_config(self, **overrides) -> FaceDetectionConfig:
        backend = self.detector_backend_var.get().strip().lower() or "yolo"
        config_values = {
            "backend": backend,
        }
        config_values.update(overrides)
        return FaceDetectionConfig(**config_values)


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
