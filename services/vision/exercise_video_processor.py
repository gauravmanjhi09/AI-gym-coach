import os
import time
import inspect
import threading
import traceback
from pathlib import Path

import cv2
import av
import numpy as np
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS

# services/vision/exercise_video_processor.py -> project root = parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml_models" / "pose_landmarker_lite.task"

# Original code mirrored the camera feed. Set False if you don't want that.
MIRROR = True


def _accepts_shape(detector):
    """True if detector.process takes (landmarks, shape), False if only (landmarks)."""
    try:
        return len(inspect.signature(detector.process).parameters) >= 2
    except (TypeError, ValueError):
        return False


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        self._frame_count = 0
        self._skip = 2            # run pose detection on every 2nd frame
        self._proc_width = 480    # detection runs on a downscaled frame
        self._last_landmarks = None
        self._last_ts_ms = 0
        self._overlay_text = None

        if not MODEL_PATH.exists():
            ml_dir = PROJECT_ROOT / "ml_models"
            raise FileNotFoundError(
                f"Pose model not found at: {MODEL_PATH} "
                f"(ml_models contents: {os.listdir(ml_dir) if ml_dir.exists() else 'folder missing'})"
            )

        base_option = python.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        # Keys MUST match EXERCISE_OPTIONS / main.py exactly
        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }
        self._takes_shape = {k: _accepts_shape(d) for k, d in self._detectors.items()}

    # ---- thread-safe accessors (kept compatible with the original API) ----
    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = dict(metrics)

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else dict(self._latest_metrics)

    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    # ---- detection ----
    def _run_detector(self, landmarks, shape):
        exercise = self.get_exercise()
        detector = self._detectors.get(exercise)
        if detector is None:
            return {"pose_detected": True}, exercise

        if self._takes_shape.get(exercise):
            metrics = detector.process(landmarks, shape)
        else:
            metrics = detector.process(landmarks)

        metrics = dict(metrics) if metrics else {}
        metrics["pose_detected"] = True
        return metrics, exercise

    # ---- drawing ----
    def _draw(self, img, landmarks):
        h, w = img.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 3)
        for p in pts:
            cv2.circle(img, p, 4, (255, 0, 0), -1)

    @staticmethod
    def _build_overlay_text(exercise, m):
        g = m.get
        if exercise == "Squats":
            return f"DEPTH: {g('depth_status', '-')}"
        if exercise == "Push-ups":
            return f"BODY: {g('body_alignment', '-')} | HIP: {g('hip_status', '-')}"
        if exercise == "Biceps Curls (Dumbbell)":
            return f"SWING: {g('swing_status', '-')}"
        if exercise == "Shoulder Press":
            return f"EXT: {g('extension_status', '-')} | BACK: {g('back_arch_status', '-')}"
        if exercise == "Lunges":
            return f"BALANCE: {g('balance_status', '-')}"
        return None

    def _draw_text(self, img, text, org):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)

    # ---- main loop ----
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if MIRROR:
            img = np.ascontiguousarray(cv2.flip(img, 1))

        try:
            self._frame_count += 1

            if self._frame_count % self._skip == 0:
                h, w = img.shape[:2]
                small = cv2.resize(img, (self._proc_width, int(h * self._proc_width / w)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts = int(time.monotonic() * 1000)
                if ts <= self._last_ts_ms:
                    ts = self._last_ts_ms + 1
                self._last_ts_ms = ts

                result = self._landmarker.detect_for_video(mp_image, ts)

                if result.pose_landmarks:
                    self._last_landmarks = result.pose_landmarks[0]
                    metrics, exercise = self._run_detector(self._last_landmarks, img.shape)
                    self._overlay_text = self._build_overlay_text(exercise, metrics)
                    self.set_latest_metrics(metrics)
                else:
                    self._last_landmarks = None
                    self._overlay_text = None
                    with self._lock:
                        if self._latest_metrics is not None:
                            self._latest_metrics["pose_detected"] = False
                        else:
                            self._latest_metrics = {"pose_detected": False}

            if self._last_landmarks:
                self._draw(img, self._last_landmarks)
                if self._overlay_text:
                    self._draw_text(img, self._overlay_text, (20, img.shape[0] - 20))
            else:
                self._draw_text(img, "NO POSE DETECTED - PLEASE FACE THE CAMERA", (20, 40))

        except Exception:
            traceback.print_exc()  # full traceback in Manage app -> logs

        return av.VideoFrame.from_ndarray(img, format="bgr24")