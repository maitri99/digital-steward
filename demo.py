"""
digital_steward/demo.py
─────────────────────────────────────────────────────────────
Live Gradio demo for the hackathon presentation.

Shows:
  • Real-time webcam or video file input
  • Annotated output frame with VIOLATION / CLEAR banner
  • Contact patch breakdown per tire (FL/FR/RL/RR)
  • FastF1 regulatory match panel
  • Live FPS counter (targeting 60 FPS on GB10)

Run
───
  python demo.py
  python demo.py --video path/to/race_clip.mp4
  python demo.py --year 2024 --event Bahrain --session Q
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from loguru import logger

from inference.engine import DigitalStewardEngine
from inference.geometry import HomographyCalibrator
from api.fastf1_validator import (
    AIViolationEvent,
    RegulatoryFilter,
    F1SessionData,
    VerdictCategory,
)


# ─────────────────────────────────────────────────────────────
# Global engine (initialised lazily on first use)
# ─────────────────────────────────────────────────────────────

_engine: Optional[DigitalStewardEngine] = None
_f1_session: Optional[F1SessionData] = None
_reg_filter: Optional[RegulatoryFilter] = None
_violation_log: list = []


def _get_engine(vit_path: str, yolo_path: str) -> DigitalStewardEngine:
    global _engine
    if _engine is None:
        logger.info("Initialising DigitalStewardEngine for demo ...")
        cal = HomographyCalibrator.default_circuit_points()
        _engine = DigitalStewardEngine(
            vit_fpn_path=vit_path,
            yolo_path=yolo_path,
            calibrator=cal,
            amp=torch.cuda.is_available(),
            use_trt=False,   # set True when TRT engine is exported
        )
    return _engine


def _load_f1_session(year: int, event: str, session: str):
    global _f1_session, _reg_filter
    try:
        _f1_session = F1SessionData(year, event, session).load()
        _reg_filter = RegulatoryFilter(_f1_session, confidence_threshold=0.80)
        return f"✅ Loaded {year} {event} {session} — {_f1_session.deleted_lap_count} deleted laps"
    except Exception as e:
        return f"⚠️ Could not load session: {e}"


# ─────────────────────────────────────────────────────────────
# Core processing function (called by Gradio)
# ─────────────────────────────────────────────────────────────

def process_frame_demo(
    frame_rgb: np.ndarray,
    vit_path: str = "models/best_model.pt",
    yolo_path: str = "yolov8n-seg.pt",
    driver_number: int = 1,
    lap_number: int = 1,
) -> Tuple[np.ndarray, str, pd.DataFrame, str]:
    """
    Called by Gradio on each input frame.

    Returns
    ───────
    (annotated_frame_rgb, verdict_text, tire_table_df, fastf1_status)
    """
    if frame_rgb is None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        return blank, "No frame", pd.DataFrame(), "—"

    engine = _get_engine(vit_path, yolo_path)

    # Gradio passes RGB; engine expects BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = engine.process_frame(frame_bgr, debug=False)

    # ── Annotated frame (back to RGB for Gradio) ──────────────
    annotated_rgb = cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)

    # ── Verdict text ──────────────────────────────────────────
    fps   = result["fps"]
    conf  = result["confidence"]
    viol  = result["violation"]

    verdict_lines = [
        f"{'🚨 VIOLATION' if viol else '✅ CLEAR'}",
        f"Confidence : {conf*100:.1f}%",
        f"FPS        : {fps:.1f}",
    ]
    if result["verdict"] and result["verdict"].note:
        verdict_lines.append(result["verdict"].note)
    verdict_text = "\n".join(verdict_lines)

    # ── Tire breakdown table ──────────────────────────────────
    tire_rows = []
    if result["verdict"] and result["verdict"].patches:
        for p in result["verdict"].patches:
            status = "OUT ❌" if p.is_fully_out else ("ON LINE ⚠️" if p.on_line_pixel_count > 0 else "IN ✅")
            tire_rows.append({
                "Tire": p.name,
                "Inside px": p.inside_pixel_count,
                "On-line px": p.on_line_pixel_count,
                "Outside px": p.out_pixel_count,
                "Status": status,
            })
    tire_df = pd.DataFrame(tire_rows) if tire_rows else pd.DataFrame(
        columns=["Tire", "Inside px", "On-line px", "Outside px", "Status"]
    )

    # ── FastF1 regulatory status ──────────────────────────────
    ff1_status = "Session not loaded."
    if _reg_filter is not None and viol and conf >= 0.80:
        event = AIViolationEvent(
            driver_number=driver_number,
            lap_number=lap_number,
            timestamp_s=time.time(),
            ai_confidence=conf,
        )
        match = _reg_filter.classify(event)
        ff1_status = f"{match.emoji()}\n{match.note}"
        _violation_log.append(match)

    return annotated_rgb, verdict_text, tire_df, ff1_status


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Digital Steward — F1 Track Limit AI",
        theme=gr.themes.Base(
            primary_hue="red",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .verdict-box { font-size: 1.4em; font-family: monospace; }
        .header { background: linear-gradient(90deg, #e10600 0%, #1a1a2e 100%);
                  padding: 20px; border-radius: 8px; }
        """,
    ) as demo:

        gr.HTML("""
        <div class='header'>
          <h1 style='color:white; margin:0; font-size:2em;'>
            🏁 Digital Steward
          </h1>
          <p style='color:#ccc; margin:4px 0 0 0;'>
            Real-time F1 Track Limit Violation Detector · Dell Pro Max GB10 · NVIDIA Blackwell
          </p>
        </div>
        """)

        with gr.Row():
            # ── Left column: inputs ───────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                cam_input = gr.Image(
                    label="Live Camera / Upload Frame",
                    sources=["webcam", "upload"],
                    type="numpy",
                    streaming=True,
                )
                with gr.Accordion("⚙️ Model Paths", open=False):
                    vit_path = gr.Textbox(
                        value="models/best_model.pt",
                        label="ViT-FPN weights",
                    )
                    yolo_path = gr.Textbox(
                        value="yolov8n-seg.pt",
                        label="YOLOv8-seg weights",
                    )

                gr.Markdown("### 📡 FastF1 Session")
                with gr.Row():
                    yr_box    = gr.Number(value=2024, label="Year",    precision=0)
                    ev_box    = gr.Textbox(value="Bahrain", label="Event")
                    ses_box   = gr.Textbox(value="Q", label="Session")
                load_btn  = gr.Button("Load Session", variant="secondary")
                ses_status = gr.Textbox(label="Session Status", interactive=False)
                load_btn.click(
                    fn=lambda y, e, s: _load_f1_session(int(y), e, s),
                    inputs=[yr_box, ev_box, ses_box],
                    outputs=ses_status,
                )

                with gr.Row():
                    drv_box = gr.Number(value=1, label="Driver #", precision=0)
                    lap_box = gr.Number(value=1, label="Lap #",    precision=0)

            # ── Right column: outputs ─────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Steward Output")
                annotated_out = gr.Image(label="Annotated Frame", type="numpy")

                with gr.Row():
                    verdict_out = gr.Textbox(
                        label="Verdict", lines=5,
                        elem_classes=["verdict-box"],
                    )
                    ff1_out = gr.Textbox(
                        label="Regulatory Status (FastF1)", lines=5,
                    )

                tire_table = gr.Dataframe(
                    label="Contact Patch Analysis",
                    headers=["Tire", "Inside px", "On-line px", "Outside px", "Status"],
                )

        # ── Streaming connection ──────────────────────────────
        cam_input.stream(
            fn=process_frame_demo,
            inputs=[cam_input, vit_path, yolo_path, drv_box, lap_box],
            outputs=[annotated_out, verdict_out, tire_table, ff1_out],
        )

        gr.Markdown("""
        ---
        **How it works:**
        1. **YOLOv8-seg** segments Track vs Out-of-Bounds zones in real time.
        2. **ViT-B/16 + FPN** detects the car and white track-limit line.
        3. **Homography** warps the broadcast view to a bird's-eye map.
        4. **Contact Patch Analysis** counts pixels per tire relative to the line.
        5. **FastF1 API** cross-validates AI decisions against FIA timing documents.

        Violation = all four tires have zero pixels on the track side of the white line.
        """)

    return demo


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int,   default=2024)
    parser.add_argument("--event",   type=str,   default="Bahrain")
    parser.add_argument("--session", type=str,   default="Q")
    parser.add_argument("--port",    type=int,   default=7860)
    parser.add_argument("--share",   action="store_true")
    args = parser.parse_args()

    # Pre-load FastF1 if args given
    if args.year and args.event:
        _load_f1_session(args.year, args.event, args.session)

    ui = build_ui()
    ui.launch(
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
