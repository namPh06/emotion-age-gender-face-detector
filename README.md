# Face Age Gender Emotion

Desktop app for realtime face detection with age, gender, and emotion prediction.

## Features

| Feature | Description |
|---|---|
| Realtime Camera | Smooth desktop webcam recognition with async inference |
| Open Image | Run recognition on an image and show an annotated result |
| Detector | Choose YOLO, Auto, or Haar face detection |
| Snapshot | Capture a still result from the webcam |
| Save Result | Save the annotated image shown in the app |
| Face Cards | Show each detected face crop with confidence bars |
| History | Review recent image and snapshot results |
| Fast/Quality Mode | Tune detection quality while keeping the default realtime behavior fast |
| Realtime Interval | Adjust background inference interval for the desktop camera |
| Stop Camera | Stop the active webcam session |
| Exit | Close the app |

## Project Structure

```text
face_age_gender_emotion/
  app.py                  # Tkinter desktop entry point
  requirements.txt
  src/
    detect_face.py        # YOLO/Haar face detection, cropping, drawing
    predict_age_gender.py # Age and gender model loading/prediction
    predict_emotion.py    # Emotion model loading/prediction
    realtime.py           # Async webcam inference and label smoothing
    utils.py              # Shared preprocessing helpers
  models/
    age_gender/
      gender_model.keras
      age_model.keras
    emotion/
      emotion_model.keras
    face_detection/
      yolo_face.pt
  notebooks/
    train_age_gender.ipynb
    train_emotion.ipynb
  datasets/
```

## Run Locally

```bash
cd face_age_gender_emotion
pip install -r requirements.txt
python scripts/download_yolo_face.py
python app.py
```

The app loads model files from `models/age_gender/` and `models/emotion/`.
It still supports the old root-level `models/*.keras` layout as a fallback.

YOLO-face is the default detector. The download script stores the Hugging Face
model `AdamCodd/YOLOv11n-face-detection` as
`models/face_detection/yolo_face.pt`. If the YOLO package or model file is
missing, the app falls back to OpenCV Haar Cascade so it can still open.

## Notes

- Webcam mode must be run on a local machine, not directly in Colab.
- Predictions are estimates and should not be used for important decisions.
- TensorFlow on native Windows may run on CPU only, so realtime labels are
  updated asynchronously to keep the webcam preview smooth.
