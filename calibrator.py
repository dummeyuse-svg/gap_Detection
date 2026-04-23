"""
calibration/calibrator.py
--------------------------
Interactive calibration tool.

Usage:
    python calibration/calibrator.py [--camera 0] [--profile board_A]
                                     [--image path/to/still.jpg]

Workflow:
  1. Live feed (or a still image) is displayed.
  2. For each hole label the user LEFT-CLICKS the centre of the expected
     gap zone.  The ROI rectangle is drawn immediately so you can verify.
  3. After all holes are clicked, press  S  to save, or  R  to redo.
  4. Press  Q  at any time to quit without saving.

The saved profile is a JSON file in data/calibration_profiles/.
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    HOLE_LABELS, ROI_HALF_WIDTH, ROI_HALF_HEIGHT,
)
from utils.image_utils import centre_to_rect, draw_roi, save_profile


# ── State shared with the mouse callback ─────────────────────────────────────
_state: dict = {}


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks: list = param["clicks"]
        labels: list = param["labels"]
        if len(clicks) < len(labels):
            idx = len(clicks)
            clicks.append({"label": labels[idx], "cx": x, "cy": y})
            print(f"  ✔ {labels[idx]} → centre ({x}, {y})")


# ── Drawing helper ────────────────────────────────────────────────────────────

def _render(base_frame: np.ndarray, clicks: list, labels: list,
            half_w: int, half_h: int) -> np.ndarray:
    """Return an annotated copy of base_frame."""
    vis = base_frame.copy()
    # Already-clicked ROIs
    for c in clicks:
        rect = centre_to_rect(c["cx"], c["cy"], half_w, half_h)
        draw_roi(vis, rect, c["label"], gap_detected=None)

    # Instruction overlay
    total  = len(labels)
    done   = len(clicks)
    remaining = labels[done] if done < total else "—"
    instructions = [
        f"Click to set ROI centre for: [{remaining}]  ({done}/{total} done)",
        "S = save   R = redo all   Q = quit",
    ]
    for i, line in enumerate(instructions):
        cv2.putText(vis, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return vis


# ── Main ─────────────────────────────────────────────────────────────────────

def run_calibration(camera_index: int, profile_name: str,
                    still_image_path: str | None,
                    hole_labels: list[str],
                    half_w: int, half_h: int) -> None:

    WINDOW = "PCB Clamp Calibrator"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, FRAME_WIDTH, FRAME_HEIGHT)

    clicks: list = []
    param = {"clicks": clicks, "labels": hole_labels}
    cv2.setMouseCallback(WINDOW, _mouse_cb, param)

    # ── Source: live camera or still image ────────────────────────────────────
    if still_image_path:
        still = cv2.imread(still_image_path)
        if still is None:
            print(f"[ERROR] Cannot read image: {still_image_path}")
            return
        base_frame = cv2.resize(still, (FRAME_WIDTH, FRAME_HEIGHT))
        cap = None
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera index {camera_index}")
            return
        base_frame = None   # will be updated in loop

    print("\n=== Calibration started ===")
    print(f"Profile name : {profile_name}")
    print(f"Hole labels  : {hole_labels}")
    print(f"ROI size     : {half_w*2} × {half_h*2} px")
    print("Click the centre of each gap zone in order shown on screen.\n")

    frozen = False   # True when user has pressed F to freeze live feed

    while True:
        # ── Grab / reuse frame ────────────────────────────────────────────────
        if cap is not None and not frozen:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera read failed – retrying…")
                continue
            base_frame = frame

        vis = _render(base_frame, clicks, hole_labels, half_w, half_h)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(30) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────────
        if key == ord('q'):
            print("[Calibration] Quit without saving.")
            break

        elif key == ord('f') and cap is not None:
            # F = freeze the live feed so clicking is easier
            frozen = not frozen
            print(f"[Calibration] Frame {'frozen' if frozen else 'live'}.")

        elif key == ord('r'):
            clicks.clear()
            print("[Calibration] Redo – all clicks cleared.")

        elif key == ord('s'):
            if len(clicks) < len(hole_labels):
                missing = hole_labels[len(clicks):]
                print(f"[WARN] Not all holes calibrated yet. Missing: {missing}")
            else:
                rois = {c["label"]: {"cx": c["cx"], "cy": c["cy"]}
                        for c in clicks}
                save_profile(profile_name, rois)
                print("[Calibration] Saved. You can close the window.")
                # Show a final confirmation frame for 2 s
                for _ in range(60):
                    conf = vis.copy()
                    cv2.putText(conf, "PROFILE SAVED  ✔", (FRAME_WIDTH//2 - 120, FRAME_HEIGHT//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow(WINDOW, conf)
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
                break

    if cap:
        cap.release()
    cv2.destroyAllWindows()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PCB Clamp – Calibration Tool")
    ap.add_argument("--camera",  type=int,  default=DEFAULT_CAMERA_INDEX,
                    help="Camera device index (default: 0)")
    ap.add_argument("--profile", type=str,  default="default",
                    help="Profile name to save (e.g. board_A)")
    ap.add_argument("--image",   type=str,  default=None,
                    help="Path to a still image instead of live camera")
    ap.add_argument("--half-w",  type=int,  default=ROI_HALF_WIDTH,
                    help=f"ROI half-width in px (default: {ROI_HALF_WIDTH})")
    ap.add_argument("--half-h",  type=int,  default=ROI_HALF_HEIGHT,
                    help=f"ROI half-height in px (default: {ROI_HALF_HEIGHT})")
    ap.add_argument("--labels",  nargs="+", default=HOLE_LABELS,
                    help="Hole labels in click order")
    args = ap.parse_args()

    run_calibration(
        camera_index    = args.camera,
        profile_name    = args.profile,
        still_image_path= args.image,
        hole_labels     = args.labels,
        half_w          = args.half_w,
        half_h          = args.half_h,
    )
