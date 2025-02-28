#!/bin/python

import os
import time

import cv2
import numpy as np
import tensorflow as tf
from prometheus_client import Gauge, start_http_server

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
g = Gauge('aeye_detected', '0 not detected, 1 detected, 2 uncertain')
c = Gauge('aeye_confidence', 'The prediction confidence')


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self, video_source: str = "", model: tf.keras.Model = None,  # type: ignore
                 frames_output_root_dir: str = "dataset",
                 tf_verbose_level: int = 0, ui: bool = False,
                 mode: str = "default", save_frames: bool = False,
                 thresholds: dict = {"uncertain": 0.65, "detected": 0.95},
                 read_all_frames: bool = False):

        if not hasattr(self, 'initialized'):
            self.video_source = video_source
            self.model = model
            self.frames_output_root_dir = frames_output_root_dir
            self.tf_verbose_level = tf_verbose_level
            self.ui = ui
            self.mode = mode
            self.save_frames = save_frames
            self.thresholds = thresholds
            self.read_all_frames = read_all_frames
            self.initialized = True


def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = frame.astype("float32") / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame


def save_frame_to_disk(frame, dir):
    frame_name = f"{int(time.time())}.jpg"

    if not os.path.exists(dir):
        os.makedirs(dir)

    file_path = os.path.join(dir, frame_name)

    if os.path.exists(file_path):
        frame_name = f"{frame_name}_{int(time.time())}.jpg"
        file_path = os.path.join(dir, frame_name)

    cv2.imwrite(file_path, frame)


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
        prediction = model.predict(processed_frame, verbose=settings.tf_verbose_level)[0][0]
        c.set(prediction)
        print(f"Aurora confidence {prediction}")
        label = "not detected"
        color = (0, 0, 255)  # Red
        active = 0
        dir = os.path.join(settings.frames_output_root_dir, "00-not-detected")

        if settings.thresholds["uncertain"] <= prediction < settings.thresholds["detected"]:
            label = "uncertain"
            active = 2
            color = (0, 255, 255)  # Yellow
            dir = os.path.join(settings.frames_output_root_dir, "uncertain")
            if settings.save_frames:
                save_frame_to_disk(frame, dir)

        if prediction >= settings.thresholds["detected"]:
            active = 1
            label = "detected"
            color = (0, 255, 0)  # Green
            dir = os.path.join(settings.frames_output_root_dir, "01-detected")
            if settings.save_frames:
                save_frame_to_disk(frame, dir)

        g.set(active)
        if settings.mode != "dynamic" and settings.save_frames is True:
            save_frame_to_disk(frame, dir)

        if settings.ui:
            cv2.putText(frame, label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("aeye", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
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

    settings.video_source = video_source
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        print("missing the model name")
        exit(1)

    model_path = os.path.join("models", str(model_name))
    settings.model = tf.keras.models.load_model(model_path)  # type: ignore

    mode = os.getenv("MODE")
    if mode is not None:
        settings.mode = mode

    print(f"running in {settings.mode} mode")
    if mode in ["testing", "classify"]:
        settings.tf_verbose_level = 1
        settings.ui = set_ui()

    if mode in ["classify", "dynamic"]:
        if os.getenv("VERBOSE"):
            settings.tf_verbose_level = 1
        settings.save_frames = True

        ford = os.getenv("FRAMES_OUTPUT_ROOT_DIR", settings.frames_output_root_dir)
        settings.frames_output_root_dir = ford

    read_all_frames = os.getenv("READ_ALL_FRAMES")
    if str(read_all_frames).lower() == "true":
        settings.read_all_frames = True

    return settings


def main():
    print("✅ Model loaded successfully!")
    cap = cv2.VideoCapture(settings.video_source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print('Could not open stream')
        exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 1)

    detect(cap, settings)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    settings = init_settings()
    start_http_server(8000)
    main()
