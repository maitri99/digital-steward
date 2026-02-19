import sys
sys.path.insert(0, '/home/dell/digital_steward')

from ultralytics import YOLO
import torch

print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

model = YOLO("/home/dell/HACKATHON/yolov8n.pt")

results = model.train(
    data="/home/dell/HACKATHON/Formula 1.v1i.yolov8/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,        # larger batch = better GPU utilization
    device=0,
    project="/home/dell/digital_steward/runs",
    name="steward_augmented",
    patience=15,
    # Our augmentation strategy
    degrees=15,      # random rotation ±15°
    fliplr=0.5,      # horizontal flip
    flipud=0.0,      # NO vertical flip (cars don't flip in F1)
    mosaic=0.5,      # mosaic augmentation
    mixup=0.2,       # mixup augmentation
    hsv_h=0.05,      # color jitter hue
    hsv_s=0.2,       # color jitter saturation
    hsv_v=0.3,       # color jitter brightness
    translate=0.1,
    scale=0.3,
    shear=2.0,
    perspective=0.0001,
    erasing=0.3,     # random erasing (simulates occlusion)
    crop_fraction=0.5,
    save=True,
    plots=True,
    verbose=True,
    workers=8,
    amp=True,        # AMP on Blackwell Tensor Cores
    cache=True,      # cache images in RAM for faster loading
)

print(f"Done! Best model: {results.save_dir}/weights/best.pt")
