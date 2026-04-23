"""
detection/gap_detector.py
--------------------------
Core gap-detection logic + a standalone CLI runner.

Detection strategy
──────────────────
Camera is placed horizontally (parallel to PCB surface) at the side.
When a hole is NOT clamped the elevated jig pin creates a thin air gap
between the jig surface and the PCB underside.  Ambient or back-light
shines through that gap → a bright horizontal streak appears in the image
at exactly the jig–PCB interface height.

For each calibrated hole we:
  1. Crop the small ROI rectangle around the expected gap location.
  2. Convert to grayscale.
  3. (Optionally) apply a gentle Gaussian blur to suppress sensor noise.
  4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) so that
     even very faint gaps become visible regardless of overall scene brightness.
  5. Threshold to find "bright" pixels.
  6. Compute the bright-pixel ratio inside the ROI.
  7. If ratio > GAP_PIXEL_RATIO_THRESHOLD → gap detected → hole NOT clamped.

Why CLAHE instead of a fixed global threshold?
  The gap produces a *relative* brightness spike, not always an absolute one.
  CLAHE normalises local contrast so a faint gap in a dim image is treated
  the same as a bright gap in a well-lit image.  The fixed threshold then
  operates on the normalised 0–255 range.

CLI usage:
    # Single-shot on a still image
    python detection/gap_detector.py --profile board_A --image test.jpg

    # Live camera loop
    python detection/gap_detector.py --profile board_A --camera 0
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    ROI_HALF_WIDTH, ROI_HALF_HEIGHT,
    BRIGHTNESS_THRESHOLD, GAP_PIXEL_RATIO_THRESHOLD,
    PRE_BLUR_KERNEL,
)
from utils.image_utils import (
    centre_to_rect, crop_roi, draw_roi,
    load_profile, list_profiles,
)


# ── CLAHE instance (built once, reused every frame) ──────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class GapDetector:
    """
    Detects whether clamping gaps are present for each calibrated hole.

    Parameters
    ----------
    profile_name : str
        Name of the calibration profile (see data/calibration_profiles/).
    half_w, half_h : int
        ROI half-dimensions in pixels.  Override here to fine-tune without
        re-running calibration.
    brightness_threshold : int
        Grayscale threshold (0–255) above which a pixel is "bright".
    gap_ratio_threshold : float
        Minimum fraction of bright pixels inside an ROI to call it a gap.
    use_clahe : bool
        Apply CLAHE for local contrast normalisation (recommended).
    """

    def __init__(self,
                 profile_name: str = "default",
                 half_w: int = ROI_HALF_WIDTH,
                 half_h: int = ROI_HALF_HEIGHT,
                 brightness_threshold: int = BRIGHTNESS_THRESHOLD,
                 gap_ratio_threshold: float = GAP_PIXEL_RATIO_THRESHOLD,
                 use_clahe: bool = True):

        self.profile_name        = profile_name
        self.half_w              = half_w
        self.half_h              = half_h
        self.brightness_threshold = brightness_threshold
        self.gap_ratio_threshold  = gap_ratio_threshold
        self.use_clahe            = use_clahe

        self._rois: dict = {}   # { label: (x1,y1,x2,y2) }
        self._load_rois()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_rois(self) -> None:
        """Load calibration profile and compute ROI rectangles."""
        profile = load_profile(self.profile_name)
        self._rois = {}
        for label, centre in profile.items():
            rect = centre_to_rect(centre["cx"], centre["cy"],
                                  self.half_w, self.half_h)
            self._rois[label] = rect
        print(f"[GapDetector] Loaded profile '{self.profile_name}' "
              f"with {len(self._rois)} ROI(s): {list(self._rois.keys())}")

    def reload_profile(self, profile_name: str | None = None) -> None:
        """Hot-reload profile (e.g. when user switches board type in UI)."""
        if profile_name:
            self.profile_name = profile_name
        self._load_rois()

    # ── Core detection ────────────────────────────────────────────────────────

    def analyse_roi(self, gray: np.ndarray,
                    rect: tuple[int, int, int, int]) -> dict:
        """
        Analyse a single ROI in a grayscale image.

        Returns
        -------
        dict with keys:
            gap_detected  : bool
            bright_ratio  : float   (0.0 – 1.0)
            bright_count  : int
            total_pixels  : int
            roi_image     : np.ndarray  (the processed ROI crop, for debugging)
        """
        patch = crop_roi(gray, rect)

        if patch.size == 0:
            return {
                "gap_detected": False,
                "bright_ratio": 0.0,
                "bright_count": 0,
                "total_pixels": 0,
                "roi_image": np.zeros((1, 1), dtype=np.uint8),
            }

        # 1. Optional blur – kills salt-and-pepper noise
        if PRE_BLUR_KERNEL != (0, 0):
            patch = cv2.GaussianBlur(patch, PRE_BLUR_KERNEL, 0)

        # 2. CLAHE – normalises local contrast so faint gaps are amplified
        if self.use_clahe:
            patch = _clahe.apply(patch)

        # 3. Threshold
        _, binary = cv2.threshold(patch, self.brightness_threshold,
                                  255, cv2.THRESH_BINARY)

        total_pixels  = binary.size
        bright_count  = int(np.count_nonzero(binary))
        bright_ratio  = bright_count / total_pixels if total_pixels > 0 else 0.0
        gap_detected  = bright_ratio >= self.gap_ratio_threshold

        return {
            "gap_detected": gap_detected,
            "bright_ratio": bright_ratio,
            "bright_count": bright_count,
            "total_pixels": total_pixels,
            "roi_image":    binary,
        }

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a full BGR (or grayscale) frame.

        Returns
        -------
        result : dict
            {
              "all_clamped": bool,          # True only if EVERY hole is OK
              "holes": {
                  "<label>": {
                      "gap_detected": bool,
                      "bright_ratio": float,
                      ...
                  },
                  ...
              }
            }
        """
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        holes = {}
        for label, rect in self._rois.items():
            holes[label] = self.analyse_roi(gray, rect)
            holes[label]["rect"] = rect   # include rect for drawing

        all_clamped = all(not h["gap_detected"] for h in holes.values())
        return {"all_clamped": all_clamped, "holes": holes}

    # ── Annotated visualisation ───────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Return a copy of *frame* with ROI boxes and status text overlaid.
        Green box  = clamped OK
        Red box    = gap detected (not clamped)
        """
        vis = frame.copy()
        for label, info in result["holes"].items():
            draw_roi(vis, info["rect"], label, info["gap_detected"])
            # Show bright ratio as a small number next to the box
            x1, y1, x2, y2 = info["rect"]
            ratio_str = f"{info['bright_ratio']*100:.1f}%"
            cv2.putText(vis, ratio_str, (x2 + 4, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # Overall status banner
        status     = "ALL CLAMPED  ✔" if result["all_clamped"] else "GAP DETECTED  ✘"
        colour     = (0, 220, 0)      if result["all_clamped"] else (0, 0, 255)
        cv2.rectangle(vis, (0, 0), (300, 32), (30, 30, 30), -1)
        cv2.putText(vis, status, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_on_image(detector: GapDetector, path: str) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read: {path}")
        return

    result = detector.detect(frame)
    vis    = detector.annotate(frame, result)

    print("\n── Detection result ──────────────────────────────────")
    print(f"  Overall status : {'PASS – all clamped' if result['all_clamped'] else 'FAIL – gap(s) detected'}")
    for label, info in result["holes"].items():
        status = "GAP  ✘" if info["gap_detected"] else "OK   ✔"
        print(f"  {label:<20} {status}  "
              f"(bright ratio = {info['bright_ratio']*100:.2f} %)")
    print("──────────────────────────────────────────────────────\n")

    cv2.imshow("Detection result – press any key", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _run_live(detector: GapDetector, camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        return

    WINDOW = "PCB Gap Detector  |  SPACE=capture  Q=quit"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, FRAME_WIDTH, FRAME_HEIGHT)

    last_result = None
    fps_t = time.time()
    frame_count = 0

    print("[Live] Running…  SPACE = trigger detection on current frame  Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        elapsed = time.time() - fps_t
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Continuously evaluate every frame so the overlays update live
        last_result = detector.detect(frame)
        vis = detector.annotate(frame, last_result)

        # FPS counter
        cv2.putText(vis, f"FPS {fps:.1f}", (FRAME_WIDTH - 100, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # Print snapshot result
            print("\n── Snapshot ──────────────────────────────────────────")
            status = "PASS" if last_result["all_clamped"] else "FAIL"
            print(f"  Status: {status}")
            for label, info in last_result["holes"].items():
                s = "GAP  ✘" if info["gap_detected"] else "OK   ✔"
                print(f"  {label:<20} {s}  "
                      f"bright={info['bright_ratio']*100:.2f}%")
            print("──────────────────────────────────────────────────────\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PCB Clamp – Gap Detector")
    ap.add_argument("--profile",   type=str,  default="default",
                    help="Calibration profile name")
    ap.add_argument("--camera",    type=int,  default=DEFAULT_CAMERA_INDEX)
    ap.add_argument("--image",     type=str,  default=None,
                    help="Run on a still image instead of live camera")
    ap.add_argument("--half-w",    type=int,  default=ROI_HALF_WIDTH)
    ap.add_argument("--half-h",    type=int,  default=ROI_HALF_HEIGHT)
    ap.add_argument("--threshold", type=int,  default=BRIGHTNESS_THRESHOLD,
                    help="Brightness threshold (0–255)")
    ap.add_argument("--ratio",     type=float, default=GAP_PIXEL_RATIO_THRESHOLD,
                    help="Minimum bright-pixel ratio to flag a gap")
    ap.add_argument("--list-profiles", action="store_true",
                    help="List saved calibration profiles and exit")
    args = ap.parse_args()

    if args.list_profiles:
        profiles = list_profiles()
        print("Saved profiles:", profiles if profiles else "(none)")
        sys.exit(0)

    detector = GapDetector(
        profile_name         = args.profile,
        half_w               = args.half_w,
        half_h               = args.half_h,
        brightness_threshold = args.threshold,
        gap_ratio_threshold  = args.ratio,
    )

    if args.image:
        _run_on_image(detector, args.image)
    else:
        _run_live(detector, args.camera)
