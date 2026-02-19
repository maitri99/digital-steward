"""
digital_steward/inference/engine.py
─────────────────────────────────────────────────────────────
Hardware Acceleration & Real-Time Inference Engine

Targets the Dell Pro Max GB10 (NVIDIA Blackwell / GB10 SoC):
  • AMP (Automatic Mixed Precision) — BF16/FP16 on Tensor Cores
  • TensorRT export for 60 FPS @ 4K
  • YOLO-seg for real-time zone segmentation
  • Memory-pinned tensors for zero-copy GPU transfer

Architecture at inference time
────────────────────────────────
  4K frame (CPU)
      │  pin_memory / non_blocking copy
      ▼
  GB10 VRAM
      │
  YOLOv8-seg (TensorRT)   ← zone segmentation (track/OOB)
      │
  ViT-FPN (TensorRT)       ← car + white-line detection
      │
  GeometricReasoner        ← homography + contact patch
      │
  RegulatoryFilter         ← FastF1 validation
      │
  Verdict + annotated frame
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from loguru import logger
from ultralytics import YOLO

from models.vit_fpn import DigitalStewardViTFPN
from inference.geometry import GeometricReasoner, HomographyCalibrator, GeometricVerdict


# ─────────────────────────────────────────────────────────────
# TensorRT export helper
# ─────────────────────────────────────────────────────────────

def export_to_tensorrt(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 720, 1280),
    workspace_gb: int = 4,
    fp16: bool = True,
) -> Path:
    """
    Export a PyTorch model to a TensorRT engine using torch-tensorrt.

    The exported engine is saved to `save_path` and can be loaded
    back via load_tensorrt_engine().

    Parameters
    ──────────
    model       – trained DigitalStewardViTFPN (or any nn.Module)
    save_path   – path to save the .ts (TorchScript + TRT) file
    input_shape – (B, C, H, W) used for tracing
    workspace_gb– TRT builder workspace in gigabytes
    fp16        – enable FP16 (recommended for Blackwell Tensor Cores)
    """
    try:
        import torch_tensorrt
    except ImportError:
        logger.warning(
            "torch-tensorrt not found. Install it to enable TRT acceleration. "
            "Falling back to standard PyTorch inference."
        )
        return None

    model.eval().cuda()
    dummy = torch.randn(*input_shape, device="cuda")

    compiled = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(shape=input_shape, dtype=torch.float16 if fp16 else torch.float32)],
        enabled_precisions={torch.float16} if fp16 else {torch.float32},
        workspace_size=workspace_gb * (1 << 30),
        truncate_long_and_double=True,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(compiled, str(save_path))
    logger.info(f"TensorRT engine saved → {save_path}")
    return save_path


def load_tensorrt_engine(path: str) -> nn.Module:
    """Load a TorchScript + TensorRT compiled model."""
    model = torch.jit.load(path)
    model.eval().cuda()
    logger.info(f"TensorRT engine loaded ← {path}")
    return model


# ─────────────────────────────────────────────────────────────
# Frame preprocessor (CPU → GPU, zero-copy)
# ─────────────────────────────────────────────────────────────

class FramePreprocessor:
    """
    Converts raw BGR numpy frames (from cv2) to normalised CUDA tensors.
    Uses pinned memory + non_blocking copies to overlap H2D transfer
    with CPU work — critical for sustained 60 FPS @ 4K.
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def __init__(self, target_h: int = 720, target_w: int = 1280):
        self.target_h = target_h
        self.target_w = target_w
        # Pinned memory buffer for fast H→D transfers
        self._pinned_buf: Optional[torch.Tensor] = None

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR numpy → normalised CUDA float16 tensor (1, 3, H, W)."""
        # Resize to model input
        frame_rgb = cv2.cvtColor(
            cv2.resize(frame_bgr, (self.target_w, self.target_h)),
            cv2.COLOR_BGR2RGB,
        )
        # Allocate or reuse pinned buffer
        if self._pinned_buf is None or self._pinned_buf.shape != (1, 3, self.target_h, self.target_w):
            self._pinned_buf = torch.empty(
                (1, 3, self.target_h, self.target_w),
                dtype=torch.float32, pin_memory=True
            )

        # Fill pinned buffer (NumPy → Tensor, still on CPU)
        arr = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0)
        self._pinned_buf[0] = arr

        # Non-blocking H→D copy (overlaps with any CPU work)
        tensor = self._pinned_buf.to("cuda", non_blocking=True)

        # Normalise in-place on GPU
        tensor = (tensor - self.MEAN) / self.STD
        return tensor.half()   # FP16 for AMP / Tensor Cores


# ─────────────────────────────────────────────────────────────
# Unified real-time inference pipeline
# ─────────────────────────────────────────────────────────────

class DigitalStewardEngine:
    """
    End-to-end inference engine for real-time track-limit monitoring.

    Integrates:
      1. YOLOv8-seg for fast track-zone segmentation
      2. ViT-FPN for precise car + line detection
      3. GeometricReasoner for homography + contact-patch analysis
      4. AMP for BF16/FP16 throughput on Blackwell Tensor Cores

    Parameters
    ──────────
    vit_fpn_path   – path to ViT-FPN weights (.pt) or TensorRT engine (.ts)
    yolo_path      – path to YOLOv8-seg weights
    calibrator     – HomographyCalibrator with circuit-specific points
    device         – "cuda" (required for GB10 acceleration)
    amp            – use Automatic Mixed Precision
    use_trt        – load the ViT-FPN as TensorRT engine (fastest)
    model_input_hw – (H, W) for the ViT-FPN model
    """

    def __init__(
        self,
        vit_fpn_path: str,
        yolo_path: str = "yolov8n-seg.pt",
        calibrator: Optional[HomographyCalibrator] = None,
        device: str = "cuda",
        amp: bool = True,
        use_trt: bool = True,
        model_input_hw: Tuple[int, int] = (720, 1280),
    ):
        self.device = torch.device(device)
        self.amp = amp

        # ── YOLOv8-seg (real-time zone segmentation) ─────────
        logger.info("Loading YOLOv8-seg ...")
        self.yolo = YOLO(yolo_path)

        # ── ViT-FPN (TRT or PyTorch) ──────────────────────────
        logger.info(f"Loading ViT-FPN from {vit_fpn_path} (TRT={use_trt}) ...")
        if use_trt and Path(vit_fpn_path).suffix == ".ts":
            self.vit_fpn = load_tensorrt_engine(vit_fpn_path)
        else:
            self.vit_fpn = DigitalStewardViTFPN(num_classes=3)
            ckpt = torch.load(vit_fpn_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.vit_fpn.load_state_dict(state, strict=False)
            self.vit_fpn.to(self.device).eval()

        # ── Preprocessor & Reasoner ───────────────────────────
        self.preprocessor = FramePreprocessor(*model_input_hw)
        self.reasoner = GeometricReasoner(calibrator=calibrator)

        # ── FPS tracking ──────────────────────────────────────
        self._frame_times: List[float] = []

        logger.info("DigitalStewardEngine ready.")

    # ── Core inference step ───────────────────────────────────

    @torch.no_grad()
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        debug: bool = False,
    ) -> Dict:
        """
        Full pipeline for a single raw camera frame.

        Returns a dict:
        {
            "violation"  : bool,
            "confidence" : float,
            "verdict"    : GeometricVerdict,
            "car_bbox"   : (x1, y1, x2, y2) in camera coords | None,
            "yolo_masks" : np.ndarray segmentation overlay | None,
            "fps"        : float,
            "annotated"  : np.ndarray BGR frame with overlays,
        }
        """
        t0 = time.perf_counter()

        result = {
            "violation": False,
            "confidence": 0.0,
            "verdict": None,
            "car_bbox": None,
            "yolo_masks": None,
            "fps": 0.0,
            "annotated": frame_bgr,
        }

        # ── Step 1: YOLOv8-seg for track/OOB zone mask ───────
        yolo_results = self.yolo(
            frame_bgr,
            conf=0.45,
            iou=0.5,
            verbose=False,
            stream=False,
        )
        yolo_seg_overlay = self._extract_yolo_overlay(yolo_results, frame_bgr)
        result["yolo_masks"] = yolo_seg_overlay

        # ── Step 2: ViT-FPN car + line detection ─────────────
        tensor = self.preprocessor.preprocess(frame_bgr)
        with autocast(enabled=self.amp, dtype=torch.float16):
            vit_out = self.vit_fpn(tensor)

        car_bbox = self._decode_car_bbox(vit_out, frame_bgr.shape)
        result["car_bbox"] = car_bbox

        # ── Step 3: Geometric reasoning ───────────────────────
        if car_bbox is not None:
            verdict = self.reasoner.reason(frame_bgr, car_bbox, debug=debug)
            result.update({
                "violation":  verdict.violation,
                "confidence": verdict.confidence,
                "verdict":    verdict,
            })

            # Annotate the frame
            annotated = self.reasoner.annotate_frame(frame_bgr, verdict, car_bbox)
            # Also overlay YOLO segmentation (semi-transparent)
            if yolo_seg_overlay is not None:
                annotated = cv2.addWeighted(annotated, 0.8, yolo_seg_overlay, 0.2, 0)
            result["annotated"] = annotated

        # ── FPS tracking ──────────────────────────────────────
        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)
        fps = 1.0 / (sum(self._frame_times) / len(self._frame_times))
        result["fps"] = fps

        return result

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:
        """
        Process an entire video file frame-by-frame.
        Returns list of per-frame result dicts.
        Optionally writes an annotated output video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

        results = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            res = self.process_frame(frame)
            res["frame_idx"] = frame_idx
            results.append(res)

            if writer is not None:
                writer.write(res["annotated"])

            if frame_idx % 100 == 0:
                logger.info(
                    f"Frame {frame_idx}  |  "
                    f"FPS={res['fps']:.1f}  |  "
                    f"{'🚨 VIOLATION' if res['violation'] else '✓ CLEAR'}"
                )
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Annotated video saved → {output_path}")

        return results

    # ── Private helpers ───────────────────────────────────────

    def _extract_yolo_overlay(self, yolo_results, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Render YOLO segmentation masks onto a copy of the frame."""
        overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
        try:
            for r in yolo_results:
                if r.masks is None:
                    continue
                for seg, cls_idx in zip(r.masks.xy, r.boxes.cls.cpu().numpy()):
                    colour = (0, 180, 0) if int(cls_idx) == 1 else (0, 0, 200)
                    pts = np.array(seg, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], colour)
        except Exception:
            pass
        return overlay

    def _decode_car_bbox(self, vit_out: Dict, frame_shape: Tuple) -> Optional[Tuple[int, int, int, int]]:
        """
        Decode the car bounding box from ViT-FPN output.
        Uses the finest FPN level (P2) classification map to find
        the cell with highest 'car' class confidence and converts
        it back to image-coordinate bounding box.

        Returns (x1, y1, x2, y2) in original frame pixel coords,
        or None if no car detected.
        """
        try:
            H_orig, W_orig = frame_shape[:2]
            # P2 is the finest scale: cls_maps[0]
            cls_map = vit_out["cls_maps"][0]      # (1, num_classes, Hf, Wf)
            reg_map = vit_out["reg_maps"][0]      # (1, 4, Hf, Wf)

            car_conf = cls_map[0, 0].softmax(0)   # class 0 = car
            Hf, Wf   = car_conf.shape

            # Find the cell with maximum car confidence
            flat_idx = car_conf.argmax()
            row = int(flat_idx // Wf)
            col = int(flat_idx % Wf)

            if float(car_conf[row, col]) < 0.3:
                return None   # No car detected above threshold

            # Decode regression: deltas in [0,1] per cell
            dx, dy, dw, dh = reg_map[0, :, row, col].tolist()
            stride_x = W_orig / Wf
            stride_y = H_orig / Hf

            cx = (col + dx) * stride_x
            cy = (row + dy) * stride_y
            bw = abs(dw) * W_orig
            bh = abs(dh) * H_orig

            x1 = int(max(0, cx - bw / 2))
            y1 = int(max(0, cy - bh / 2))
            x2 = int(min(W_orig - 1, cx + bw / 2))
            y2 = int(min(H_orig - 1, cy + bh / 2))

            if x2 - x1 < 10 or y2 - y1 < 10:
                return None

            return (x1, y1, x2, y2)
        except Exception as exc:
            logger.debug(f"BBox decode failed: {exc}")
            return None
