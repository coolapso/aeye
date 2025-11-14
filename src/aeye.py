#!/bin/python

import os
import time

import cv2
import numpy as np
import tensorflow as tf
from prometheus_client import Gauge, start_http_server
from tensorflow.keras.applications import mobilenet_v2, efficientnet

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
c = Gauge(
    "aeye_detection_confidence", "The prediction confidence for a detected class", ["class_name"]
)


class Settings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        video_source: str = "",
        model: tf.keras.Model = None,  # type: ignore
        tf_verbose_level: int = 0,
        ui: bool = False,
        mode: str = "default",
        confidence_threshold: float = 0.75,
        class_labels: list = [],
        read_all_frames: bool = False,
        base_model_name: str = "efficientnet",
    ):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.video_source = video_source
        self.model = model
        self.tf_verbose_level = tf_verbose_level
        self.ui = ui
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.read_all_frames = read_all_frames
        self.base_model_name = base_model_name
        self.initialized = True


def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    if settings.base_model_name == "mobilenet":
        frame = mobilenet_v2.preprocess_input(frame)
    elif settings.base_model_name == "efficientnet":
        frame = efficientnet.preprocess_input(frame)
    else:
        frame = frame.astype("float32") / 255.0  # Normalize for custom model
    return frame


def set_ui() -> bool:
    session = os.getenv("XDG_SESSION_TYPE")
    if session in ["x11", "wayland"]:
        return True

    if session == "tty":
        print("running on terminal session, not showing ui")

    return False


def detect(cap: cv2.VideoCapture, settings: Settings):
    model = settings.model
    last_processed = time.time()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    rtsp = True
    if "rtsp" not in settings.video_source:
        rtsp = False

    while cap.isOpened():
        # Process one frame per second to handle the feed in real time and prevent
        # TensorFlow from buffering frames, which can skew metrics between the metric
        # time and the actual frame timestamp. Use cap.grab() as it's less resource
        # intensive than cap.read().
        if rtsp and not settings.read_all_frames:
            if not time.time() - last_processed > 1:
                cap.grab()
                continue

        if not rtsp and not settings.read_all_frames:
            if frame_count % fps != 0:
                cap.grab()
                frame_count += 1
                continue

        ready, frame = cap.read()
        if not ready:
            print("End of stream or failed to read frame.")
            break

        processed_frame = preprocess_frame(frame)
        batch_predictions = model.predict(processed_frame, verbose=settings.tf_verbose_level)
        predictions = np.array(batch_predictions[0])

        for i, confidence in enumerate(predictions):
            class_name = settings.class_labels[i]
            c.labels(class_name=class_name).set(confidence)

        predicted_class_index = np.argmax(predictions)
        confidence = predictions[predicted_class_index]

        predicted_class_name = settings.class_labels[predicted_class_index]
        label = f"{predicted_class_name} ({confidence:.2f})"

        if confidence >= settings.confidence_threshold:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.5 and confidence < 0.75:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        if settings.ui:
            cv2.putText(frame, label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("aeye", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                break

        last_processed = time.time()
        frame_count += 1


def init_settings() -> Settings:
    settings = Settings()

    video_source = os.getenv("VIDEO_SOURCE")
    if not video_source:
        print("Missing video source, please set the VIDEO_SOURCE env variable")
        exit(1)

    mode = os.getenv("MODE", "default")
    settings.mode = mode

    print(f"running in {settings.mode} mode")
    if mode in "testing":
        settings.tf_verbose_level = 1
        settings.ui = set_ui()

    settings.video_source = video_source
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        print("missing the model name")
        exit(1)

    model_path = os.path.join("models", str(model_name))
    settings.model = tf.keras.models.load_model(model_path)  # type: ignore

    base_model_name = os.getenv("BASE_MODEL_NAME")
    if base_model_name:
        settings.base_model_name = base_model_name

    class_labels_str = os.getenv("CLASS_LABELS")
    if not class_labels_str:
        print("Missing class labels, please set the CLASS_LABELS env variable (comma-separated)")
        exit(1)
    settings.class_labels = [label.strip() for label in class_labels_str.split(",")]
    print(f"Loaded class labels: {settings.class_labels}")

    try:
        threshold = os.getenv("CONFIDENCE_THRESHOLD")
        if threshold:
            settings.confidence_threshold = float(threshold)
    except ValueError:
        print(f"Invalid CONFIDENCE_THRESHOLD value. Using default: {settings.confidence_threshold}")

    mode = os.getenv("MODE")
    if mode is not None:
        settings.mode = mode

    read_all_frames = os.getenv("READ_ALL_FRAMES")
    if str(read_all_frames).lower() == "true":
        settings.read_all_frames = True

    return settings


def main():
    print("✅ Model loaded successfully!")
    cap = cv2.VideoCapture(settings.video_source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Could not open stream")
        exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 1)

    detect(cap, settings)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    settings = init_settings()
    start_http_server(port=8000, addr="0.0.0.0")
    main()
