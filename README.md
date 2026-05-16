# Face Age Gender Emotion

Desktop app for realtime face detection with age, gender, and emotion prediction.

## Features

| Feature | Description |
|---|---|
| Start Camera | Realtime webcam recognition with FPS counter |
| Open Image | Run recognition on an image and save an annotated result |
| Stop Camera | Stop the active webcam session |
| Exit | Close the app |

## Project Structure

```text
face_age_gender_emotion/
  app.py                  # GUI entry point
  requirements.txt
  src/
    detect_face.py        # Haar Cascade face detection, cropping, drawing
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
  notebooks/
    train_age_gender.ipynb
    train_emotion.ipynb
  datasets/
```

## Run Locally

```bash
cd face_age_gender_emotion
pip install -r requirements.txt
python app.py
```

The app loads model files from `models/age_gender/` and `models/emotion/`.
It still supports the old root-level `models/*.keras` layout as a fallback.

## Notes

- Webcam mode must be run on a local machine, not directly in Colab.
- Predictions are estimates and should not be used for important decisions.
- TensorFlow on native Windows may run on CPU only, so realtime labels are
  updated asynchronously to keep the webcam preview smooth.
