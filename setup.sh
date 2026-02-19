#!/usr/bin/env bash
# ============================================================
#  Digital Steward — Quick-Start Setup Script
#  Run this once after connecting your VS Code to the GB10.
# ============================================================

set -e
echo "================================================="
echo "  Digital Steward — GB10 Environment Setup"
echo "================================================="

# ── 1. Python virtual environment ────────────────────────────
echo "[1/6] Creating virtual environment ..."
python3 -m venv .venv
source .venv/bin/activate

# ── 2. Upgrade pip ───────────────────────────────────────────
echo "[2/6] Upgrading pip ..."
pip install --upgrade pip wheel setuptools

# ── 3. Install PyTorch (Blackwell / CUDA 12.4) ───────────────
echo "[3/6] Installing PyTorch for CUDA 12.4 (GB10) ..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ── 4. Install project dependencies ─────────────────────────
echo "[4/6] Installing project requirements ..."
pip install -r requirements.txt

# ── 5. Download YOLOv8n-seg weights ─────────────────────────
echo "[5/6] Downloading YOLOv8n-seg weights ..."
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"

# ── 6. Create FastF1 cache dir ───────────────────────────────
echo "[6/6] Creating FastF1 cache directory ..."
mkdir -p .fastf1_cache
mkdir -p models outputs logs data/raw/images/train data/raw/labels/train
mkdir -p data/raw/images/val data/raw/labels/val
mkdir -p data/raw/images/test data/raw/labels/test

echo ""
echo "================================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Place your dataset images in data/raw/images/{train,val,test}/"
echo "     and YOLO labels in data/raw/labels/{train,val,test}/"
echo ""
echo "  2. Train the model:"
echo "     python train.py --config configs/config.yaml"
echo ""
echo "  3. Run the live demo:"
echo "     python demo.py --year 2024 --event Bahrain --session Q"
echo ""
echo "  4. Export to TensorRT (after training):"
echo "     python -c \""
echo "       from inference.engine import export_to_tensorrt"
echo "       from models.vit_fpn import DigitalStewardViTFPN"
echo "       import torch"
echo "       m = DigitalStewardViTFPN()"
echo "       m.load_state_dict(torch.load('models/best_model.pt')['model_state_dict'])"
echo "       export_to_tensorrt(m, 'models/steward_trt.ts')"
echo "     \""
echo "================================================="
