from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

model = YOLO("/home/dell/HACKATHON/yolov8n.pt")

results = model.train(
    data="/home/dell/HACKATHON/Formula 1.v1i.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    project="/home/dell/digital_steward/runs",
    name="steward_v1",
    patience=10,
    augment=True,
    degrees=15,
    fliplr=0.5,
    mosaic=0.5,
    mixup=0.2,
    hsv_h=0.05,
    hsv_s=0.2,
    hsv_v=0.3,
    save=True,
    plots=True,
    verbose=True,
)

print("Training complete!")
print(f"Best model saved to: {results.save_dir}")
