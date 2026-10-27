import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def main():
    if not MODEL_PATH.exists():
        print("Downloading the Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    try:
        with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                height, width = frame.shape[:2]
                for face_landmarks in results.face_landmarks:
                    for landmark in face_landmarks:
                        point = (int(landmark.x * width), int(landmark.y * height))
                        cv2.circle(frame, point, 1, (0, 255, 0), -1)

                cv2.imshow("MediaPipe Face Landmarker", frame)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
