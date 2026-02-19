"""
digital_steward/inference/geometry.py
─────────────────────────────────────────────────────────────
Geometric Reasoning Layer — the "Ground Truth" rulings engine.

Pipeline
─────────
1.  Perspective Transform (Homography)
    Warps a skewed broadcast camera feed into a top-down
    bird's-eye view using cv2.getPerspectiveTransform.

2.  Contact Patch Analysis
    Locates the X-Y pixel coordinates of all four tire
    contact patches in the warped (flat) image.

3.  Violation Logic
    A FIA-compliant ruling: a track-limit violation occurs
    when ALL FOUR wheels have crossed the white boundary line,
    i.e. not a single pixel of any tire remains on or inside
    the line.

Why top-down view?
    In a perspective-projected camera frame, the boundary line
    appears tapered and curved.  After homography, the line
    becomes a straight horizontal/vertical strip of constant
    width, making "inside" pixel counting trivially accurate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class TireContactPatch:
    """Pixel-level contact patch of a single tire in the warped top-down view."""
    name: str                      # "FL" | "FR" | "RL" | "RR"
    center_xy: Tuple[int, int]     # (col, row) in warped image
    radius_px: int = 12
    inside_pixel_count: int = 0    # pixels on the track side of the white line
    on_line_pixel_count: int = 0   # pixels landing exactly on the white line
    out_pixel_count: int = 0       # pixels outside the track boundary

    @property
    def is_fully_out(self) -> bool:
        """True when the tire has zero contact with the track or the line."""
        return self.inside_pixel_count == 0 and self.on_line_pixel_count == 0


@dataclass
class GeometricVerdict:
    """Output of the GeometricReasoner for one frame."""
    violation: bool
    confidence: float                          # 0.0 – 1.0
    patches: List[TireContactPatch] = field(default_factory=list)
    homography_matrix: Optional[np.ndarray] = None
    warped_frame: Optional[np.ndarray] = None  # bird's-eye BGR image (debug)
    note: str = ""


# ─────────────────────────────────────────────────────────────
# Colour masks for the white boundary line
# ─────────────────────────────────────────────────────────────

# These HSV thresholds pick up the bright white painted line
# under typical circuit lighting. They are tunable per-circuit.
WHITE_LINE_HSV_LOWER = np.array([0, 0, 200], dtype=np.uint8)
WHITE_LINE_HSV_UPPER = np.array([180, 30, 255], dtype=np.uint8)

# "In-bounds" green grass / tarmac (circuit interior) is not white;
# we use a simple thresholded inverse of the line mask.


# ─────────────────────────────────────────────────────────────
# Homography calibrator
# ─────────────────────────────────────────────────────────────

class HomographyCalibrator:
    """
    Stores a set of 4 source points (broadcast view) and 4
    destination points (flat top-down canonical view), and
    computes the 3×3 perspective transform matrix.

    Usage
    ─────
    cal = HomographyCalibrator()
    cal.set_points(
        src=[(x0,y0),(x1,y1),(x2,y2),(x3,y3)],   # corners in camera frame
        dst=[(0,0),(W,0),(W,H),(0,H)],             # rectangular canvas
    )
    warped = cal.warp(frame_bgr)
    """

    def __init__(self):
        self.M: Optional[np.ndarray] = None     # 3×3 homography matrix
        self.M_inv: Optional[np.ndarray] = None  # inverse (flat → camera)
        self.dst_size: Tuple[int, int] = (1280, 720)  # (W, H) of output canvas

    def set_points(
        self,
        src: List[Tuple[float, float]],
        dst: List[Tuple[float, float]],
        dst_size: Tuple[int, int] = (1280, 720),
    ):
        """
        Compute the homography from 4 point correspondences.

        src  – 4 points in the original camera image (ordered TL, TR, BR, BL)
        dst  – 4 points in the top-down canvas
        """
        if len(src) != 4 or len(dst) != 4:
            raise ValueError("Exactly 4 source and 4 destination points required.")

        src_np = np.array(src, dtype=np.float32)
        dst_np = np.array(dst, dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_np, dst_np)
        self.M_inv = cv2.getPerspectiveTransform(dst_np, src_np)
        self.dst_size = dst_size

    def warp(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply perspective warp; returns the bird's-eye BGR image."""
        if self.M is None:
            raise RuntimeError("Call set_points() before warp().")
        return cv2.warpPerspective(frame_bgr, self.M, self.dst_size)

    def unproject_point(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Map a point from the warped canvas back to the camera frame."""
        if self.M_inv is None:
            raise RuntimeError("Call set_points() before unproject_point().")
        pt_h = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt_h, self.M_inv)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    @classmethod
    def default_circuit_points(cls, frame_w: int = 3840, frame_h: int = 2160):
        """
        Returns a HomographyCalibrator pre-loaded with example points
        for a generic circuit hairpin corner broadcast camera.
        These MUST be replaced with real hand-labelled calibration points
        for each circuit and camera position before production use.
        """
        cal = cls()
        # Typical broadcast hairpin: white line runs roughly across the bottom third
        src = [
            (frame_w * 0.25, frame_h * 0.55),  # TL of track-limit region
            (frame_w * 0.75, frame_h * 0.55),  # TR
            (frame_w * 0.90, frame_h * 0.90),  # BR
            (frame_w * 0.10, frame_h * 0.90),  # BL
        ]
        dst_w, dst_h = 1280, 720
        dst = [(0, 0), (dst_w, 0), (dst_w, dst_h), (0, dst_h)]
        cal.set_points(src, dst, (dst_w, dst_h))
        return cal


# ─────────────────────────────────────────────────────────────
# Contact patch extractor
# ─────────────────────────────────────────────────────────────

class ContactPatchExtractor:
    """
    Given the bounding box of an F1 car in the warped top-down view,
    estimates all four tire contact patch centres and samples pixels
    around each one to count inside / on-line / outside pixels.
    """

    # Relative offsets of tire centres within the car bounding box
    # (normalised: 0 = left/top edge, 1 = right/bottom edge)
    TIRE_OFFSETS = {
        "FL": (0.15, 0.10),   # Front-Left  (col_frac, row_frac)
        "FR": (0.85, 0.10),   # Front-Right
        "RL": (0.15, 0.90),   # Rear-Left
        "RR": (0.85, 0.90),   # Rear-Right
    }

    def __init__(self, radius_px: int = 12):
        self.radius_px = radius_px

    def extract(
        self,
        warped_bgr: np.ndarray,
        car_bbox_xyxy: Tuple[int, int, int, int],
        line_mask: np.ndarray,   # binary mask: 255 = white line, 0 = other
    ) -> List[TireContactPatch]:
        """
        warped_bgr      – bird's-eye BGR frame
        car_bbox_xyxy   – (x1, y1, x2, y2) of the car in the warped frame
        line_mask       – single-channel binary mask (0/255) of the white line

        Returns list of 4 TireContactPatch objects.
        """
        x1, y1, x2, y2 = car_bbox_xyxy
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

        h, w = warped_bgr.shape[:2]

        # "In-bounds" mask = everything that is NOT the white line
        # and on the track side (below / inside the line)
        # Simple approach: in-bounds = NOT white line AND inside a manually
        # defined polygon for the track interior.
        # For the MVP we use pixel intensity: dark tarmac pixels.
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        # Tarmac is generally dark (0-100 intensity) vs. grass (50-150) vs. white line (>200)
        tarmac_mask = (gray < 100).astype(np.uint8) * 255
        outside_mask = ((gray > 150) & (line_mask == 0)).astype(np.uint8) * 255

        patches = []
        for name, (cf, rf) in self.TIRE_OFFSETS.items():
            cx = int(x1 + cf * bw)
            cy = int(y1 + rf * bh)

            # Clamp to image bounds
            cx = max(self.radius_px, min(w - self.radius_px - 1, cx))
            cy = max(self.radius_px, min(h - self.radius_px - 1, cy))

            # Sample a disk of radius_px around the contact patch centre
            disk_mask = self._disk_mask(cy, cx, self.radius_px, h, w)

            inside_px = int(np.sum((tarmac_mask > 0) & disk_mask))
            on_line_px = int(np.sum((line_mask > 0) & disk_mask))
            out_px = int(np.sum((outside_mask > 0) & disk_mask))

            patches.append(TireContactPatch(
                name=name,
                center_xy=(cx, cy),
                radius_px=self.radius_px,
                inside_pixel_count=inside_px,
                on_line_pixel_count=on_line_px,
                out_pixel_count=out_px,
            ))

        return patches

    @staticmethod
    def _disk_mask(cy: int, cx: int, r: int, h: int, w: int) -> np.ndarray:
        """Returns a boolean array of shape (h, w) True inside the disk."""
        Y, X = np.ogrid[:h, :w]
        return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


# ─────────────────────────────────────────────────────────────
# Geometric Reasoner — main entry point
# ─────────────────────────────────────────────────────────────

class GeometricReasoner:
    """
    Orchestrates homography → contact patch extraction → violation logic.

    Usage
    ─────
    reasoner = GeometricReasoner(calibrator=my_cal)
    verdict = reasoner.reason(
        frame_bgr=raw_4k_frame,
        car_bbox_camera=(x1, y1, x2, y2),   # detection output in camera coords
    )
    if verdict.violation:
        print(f"VIOLATION  confidence={verdict.confidence:.2f}")
    """

    def __init__(
        self,
        calibrator: Optional[HomographyCalibrator] = None,
        tire_radius_px: int = 12,
    ):
        self.calibrator = calibrator or HomographyCalibrator.default_circuit_points()
        self.patch_extractor = ContactPatchExtractor(radius_px=tire_radius_px)

    def _build_line_mask(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Threshold the warped image to isolate the white boundary line."""
        hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_LINE_HSV_LOWER, WHITE_LINE_HSV_UPPER)
        # Morphological clean-up to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def reason(
        self,
        frame_bgr: np.ndarray,
        car_bbox_camera: Tuple[int, int, int, int],
        debug: bool = False,
    ) -> GeometricVerdict:
        """
        Parameters
        ──────────
        frame_bgr        – raw BGR frame from broadcast camera (any resolution)
        car_bbox_camera  – (x1, y1, x2, y2) of the car in camera pixel coords
        debug            – if True, include warped frame in verdict for visualisation

        Returns
        ───────
        GeometricVerdict with:
          .violation  – bool (FIA rule: all 4 tires fully past the line)
          .confidence – fraction of tires that are fully outside [0.0, 1.0]
          .patches    – list of TireContactPatch objects
        """
        # 1. Warp to top-down view
        warped = self.calibrator.warp(frame_bgr)
        M = self.calibrator.M
        W, H = self.calibrator.dst_size

        # 2. Project car bounding box into warped space
        x1c, y1c, x2c, y2c = car_bbox_camera
        corners_cam = np.array([
            [[x1c, y1c]], [[x2c, y1c]], [[x2c, y2c]], [[x1c, y2c]]
        ], dtype=np.float32)
        corners_warp = cv2.perspectiveTransform(corners_cam, M)
        wx = corners_warp[:, 0, 0]
        wy = corners_warp[:, 0, 1]
        bbox_warp = (int(wx.min()), int(wy.min()), int(wx.max()), int(wy.max()))

        # 3. Build white-line mask in warped space
        line_mask = self._build_line_mask(warped)

        # 4. Extract contact patches
        patches = self.patch_extractor.extract(warped, bbox_warp, line_mask)

        # 5. Violation logic (FIA sporting regulation §27.3 interpretation)
        #    Violation iff ALL four tires are fully beyond the white line
        #    (inside_pixel_count == 0  AND  on_line_pixel_count == 0)
        fully_out = [p.is_fully_out for p in patches]
        n_out = sum(fully_out)
        violation = all(fully_out)

        # Confidence = fraction of tires that are fully outside
        confidence = n_out / 4.0

        note = (
            "All 4 tires fully beyond track limit." if violation
            else f"{n_out}/4 tires outside — no violation yet."
        )

        verdict = GeometricVerdict(
            violation=violation,
            confidence=confidence,
            patches=patches,
            homography_matrix=M,
            warped_frame=warped if debug else None,
            note=note,
        )
        return verdict

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        verdict: GeometricVerdict,
        car_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Draw bounding box and verdict banner on the original camera frame.
        Returns annotated BGR image.
        """
        out = frame_bgr.copy()
        x1, y1, x2, y2 = car_bbox

        colour = (0, 0, 255) if verdict.violation else (0, 220, 60)
        thickness = 3
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

        label = "VIOLATION" if verdict.violation else f"CLEAR {verdict.confidence*100:.0f}%"
        cv2.putText(
            out, label,
            (x1, max(y1 - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4, colour, 3, cv2.LINE_AA,
        )
        return out
