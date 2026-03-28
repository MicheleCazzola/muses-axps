import os
import torch

# ==========================================================
# CONFIG
# ==========================================================

# Define image size (H, W) and resize dimensions for the model
BASE_SIZE = (1080, 1920)
RESIZE_MODEL = (512, 1024)

DATA_ROOT = os.path.join("data", "muses")
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
BATCH_SIZE = 2

# Use more workers for CPU/MPS
# Set to 0 for CUDA to avoid potential issues with multiprocessing on some platforms (e.g. Colab)
NUM_WORKERS = 2 if DEVICE != "cuda" else 0      