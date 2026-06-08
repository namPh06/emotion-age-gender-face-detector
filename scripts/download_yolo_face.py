"""Download the YOLO-face model used by the desktop app."""
from __future__ import annotations

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIR / "models" / "face_detection" / "yolo_face.pt"
REPO_ID = "AdamCodd/YOLOv11n-face-detection"
FILENAME = "model.pt"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cached_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    shutil.copyfile(cached_path, OUTPUT_PATH)
    print(f"Saved YOLO-face model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
