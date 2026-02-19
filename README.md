# 🏁 Digital Steward

**Real-time F1 Track Limit Violation Detector**  
NYU CDS × Dell Technologies × NVIDIA Hackathon 2026

## What it does
Automatically detects Formula 1 track limit violations from broadcast footage in **3.2ms per frame**, running entirely on a Dell Pro Max GB10 (NVIDIA Blackwell). Validated against 49 official FIA steward decisions from the 2023 Austrian GP via the FastF1 API.

## Results
| Metric | Value |
|--------|-------|
| Penalty mAP50 | 89.5% |
| Penalty Precision | **100%** |
| Overall mAP50 | 92.3% |
| Inference speed | 3.2ms/frame |
| Training time on GB10 | 3 minutes |

## Hardware
- Dell Pro Max GB10
- NVIDIA Blackwell GPU
- 122GB Unified Memory
- CUDA 13.0 / PyTorch 2.10

## Run the demo
```bash
source /path/to/f1_steward_env/bin/activate
python3 demo_simple.py
# Open http://localhost:7860
```

## Key files
- `demo_simple.py` — Gradio demo (4 tabs: detection, ground truth, FIA validation, model performance)
- `train_augmented.py` — YOLOv8 training script with custom augmentations
- `models/vit_fpn.py` — ViT-FPN architecture (production roadmap)
- `data/dataset.py` — Dataset loader with augmentation pipeline

## Demo tabs
1. **Live Detection** — Upload F1 frame, get instant penalty/clear/manual-review verdict
2. **Ground Truth Comparison** — Side-by-side vs human labels on 15 held-out test images
3. **FIA Steward Validation** — 49 real deletion events from 2023 Austrian GP via FastF1 API
4. **Model Performance** — Confusion matrix, F1 curve, PR curve, training loss plots
