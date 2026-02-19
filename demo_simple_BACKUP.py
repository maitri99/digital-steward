import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("/home/dell/digital_steward/runs/steward_v1/weights/best.pt")

def predict(image):
    results = model(image, conf=0.45, device=0)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    verdict = "✅ NO VIOLATION"
    confidence = 0.0
    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls == 1 and conf > confidence:  # Penalty class
            confidence = conf
            verdict = f"🚨 PENALTY DETECTED ({conf*100:.1f}% confidence)"
    
    return annotated_rgb, verdict

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload F1 Frame"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Steward Verdict", elem_classes=["verdict"]),
    ],
    title="🏁 Digital Steward — F1 Track Limit Detector",
    description="Upload an F1 broadcast frame to detect track limit violations. Running on Dell Pro Max GB10 with NVIDIA Blackwell.",
)

demo.launch(server_port=7860, share=False)
