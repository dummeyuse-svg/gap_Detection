"""
utils/image_utils.py
--------------------
Shared helpers used by both calibration and detection modules.
"""

import json
import os
import sys
import cv2
import numpy as np

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    ROI_HALF_WIDTH, ROI_HALF_HEIGHT,
    BRIGHTNESS_THRESHOLD, GAP_PIXEL_RATIO_THRESHOLD,
    PRE_BLUR_KERNEL, CALIBRATION_PROFILES_DIR,
)


# ── ROI helpers ───────────────────────────────────────────────────────────────

def centre_to_rect(cx: int, cy: int,
                   half_w: int = ROI_HALF_WIDTH,
                   half_h: int = ROI_HALF_HEIGHT) -> tuple[int, int, int, int]:
    """Convert a centre point to (x1, y1, x2, y2) rectangle."""
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def crop_roi(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Safely crop an ROI from an image; returns empty array if out of bounds."""
    x1, y1, x2, y2 = rect
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return image[y1:y2, x1:x2]


def draw_roi(frame: np.ndarray,
             rect: tuple[int, int, int, int],
             label: str,
             gap_detected: bool | None = None) -> None:
    """
    Draw an ROI rectangle on *frame* (in-place).
    Colour:
      - Yellow  → not yet evaluated (calibration view)
      - Green   → clamped OK
      - Red     → gap detected (not clamped)
    """
    x1, y1, x2, y2 = rect
    if gap_detected is None:
        colour = (0, 220, 220)   # yellow
    elif gap_detected:
        colour = (0, 0, 255)     # red  – problem
    else:
        colour = (0, 200, 0)     # green – OK

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
    cv2.putText(frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


# ── Profile persistence ───────────────────────────────────────────────────────

def profile_path(profile_name: str) -> str:
    os.makedirs(CALIBRATION_PROFILES_DIR, exist_ok=True)
    return os.path.join(CALIBRATION_PROFILES_DIR, f"{profile_name}.json")


def save_profile(profile_name: str, rois: dict) -> None:
    """
    Save ROI centres to a JSON profile.

    rois: { "hole_front": {"cx": 340, "cy": 210}, "hole_rear": {...} }
    """
    path = profile_path(profile_name)
    with open(path, "w") as f:
        json.dump(rois, f, indent=2)
    print(f"[Profile] Saved → {path}")


def load_profile(profile_name: str) -> dict:
    """Load a saved profile; raises FileNotFoundError if missing."""
    path = profile_path(profile_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No calibration profile '{profile_name}' found at {path}.\n"
            "Run calibration/calibrator.py first."
        )
    with open(path) as f:
        return json.load(f)


def list_profiles() -> list[str]:
    os.makedirs(CALIBRATION_PROFILES_DIR, exist_ok=True)
    return [f[:-5] for f in os.listdir(CALIBRATION_PROFILES_DIR)
            if f.endswith(".json")]
