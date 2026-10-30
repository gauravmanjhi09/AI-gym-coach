import os
import cv2
import av
import numpy as np
import mediapipe as mp
import threading
from pathlib import Path
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
MODEL_PATH = PROJECT_ROOT / "ml_models" / "pose_landmarker_full.task"

class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pose model not found at: {MODEL_PATH} "
                f"(ml_models contents: "
                f"{os.listdir(PROJECT_ROOT / 'ml_models') if (PROJECT_ROOT / 'ml_models').exists() else 'folder missing'})"
            )

        base_option = python.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        # ... baaki code (self._detectors, self._frame_timestamps_ms, aur saare methods) waisa hi rahega