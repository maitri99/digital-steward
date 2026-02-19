# Digital Steward — F1 Track Limit Violation Detector
## Built for the NYU CDS × Dell Technologies × NVIDIA Hackathon

---

### What it does
**The Digital Steward** is a real-time, computer vision-based referee for Formula 1 racing that automatically detects track limit violations — when all four wheels of a car cross the white boundary line.

One race generated **1,200+ potential violations**, causing a **5-hour delay** in final results. The Digital Steward makes that decision **instant**.

---

### Architecture

```
4K Broadcast Frame
        │
        ├──► YOLOv8n-seg (TensorRT)
        │         Track vs Out-of-Bounds segmentation
        │
        ├──► ViT-B/16 + FPN (TensorRT)
        │         Car + white-line detection
        │         (Global context via Self-Attention)
        │
        ├──► Geometric Reasoner
        │         Homography → bird's-eye view
        │         Contact patch analysis (4 tires)
        │         FIA rule: all 4 tires past line = violation
        │
        └──► Regulatory Filter (FastF1 API)
                  MATCH   ✅ — AI ∩ FIA both flagged
                  WARNING ⚠️ — FIA flagged, AI missed (occlusion?)
                  FLAG    🚩 — AI flagged, FIA missed (human error?)
                  CLEAR   ✓  — No violation
```

---

### Setup (on GB10)

```bash
chmod +x setup.sh && ./setup.sh
```

---

### Training

```bash
python train.py --config configs/config.yaml
```

Features:
- AMP (BF16/FP16) on Blackwell Tensor Cores
- ViT backbone frozen for first 5 epochs, then gradually unfrozen
- Mosaic + MixUp + Rain + Night augmentations
- Cosine LR decay with warm-up
- Early stopping + best-model checkpointing

---

### Live Demo

```bash
python demo.py --year 2024 --event Bahrain --session Q
```

Opens a Gradio UI on `http://localhost:7860` with:
- Webcam / video file input
- Real-time annotated output (VIOLATION / CLEAR)
- Per-tire contact patch breakdown
- FastF1 regulatory match panel

---

### Export to TensorRT (60 FPS @ 4K)

```python
from inference.engine import export_to_tensorrt
from models.vit_fpn import DigitalStewardViTFPN
import torch

model = DigitalStewardViTFPN()
model.load_state_dict(torch.load("models/best_model.pt")["model_state_dict"])
export_to_tensorrt(model, "models/steward_trt.ts", fp16=True)
```

---

### Dataset Format

```
data/raw/
    images/
        train/  *.jpg
        val/
        test/
    labels/          # YOLO format: class cx cy w h (normalised)
        train/  *.txt
        val/
        test/
```

Classes: `0=car`, `1=track`, `2=out_of_bounds`

---

### Files

| File | Description |
|------|-------------|
| `configs/config.yaml` | All hyperparameters |
| `data/augmentation.py` | Rain, night, mosaic, mixup, torchvision v2 |
| `data/dataset.py` | YOLO-format dataset + collate |
| `models/vit_fpn.py` | ViT-B/16 + FPN + detection/seg heads |
| `inference/geometry.py` | Homography + contact patch + violation logic |
| `inference/engine.py` | TensorRT pipeline + AMP + 60 FPS loop |
| `api/fastf1_validator.py` | FastF1 regulatory truth layer |
| `train.py` | Full training loop |
| `demo.py` | Gradio live demo |

---

### Judging Criteria Alignment

| Criterion | How this project addresses it |
|-----------|-------------------------------|
| **Technology** | All inference on GB10: ViT-FPN + YOLO both TensorRT-exported, AMP enabled, tensors pinned to CUDA |
| **Efficiency/Scalability** | 60 FPS @ 4K via TensorRT; 2× GB10 units can be stacked via ConnectX for dual-angle coverage |
| **Practicality/Impact** | Eliminates 5-hour stewarding delays; FLAG category surfaces FIA human errors |
| **Presentation** | Gradio UI with live camera feed, per-tire breakdown, regulatory match panel |
