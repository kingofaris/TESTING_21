# backend/config.py
import os
from pathlib import Path

# Path
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_CACHE_DIR = BASE_DIR / "models"

# Pastikan folder ada
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Model - pakai versi ringan untuk 4GB VRAM
# LTX Video 2B (versi kecil, cocok untuk VRAM terbatas)
MODEL_ID = "Lightricks/LTX-Video"

# Setting untuk 4GB VRAM
DEVICE = "cuda"
DTYPE = "bfloat16"          # hemat VRAM vs float32
ENABLE_CPU_OFFLOAD = True   # WAJIB untuk 4GB
ENABLE_VAE_SLICING = True   # potong VAE jadi bagian kecil
ENABLE_VAE_TILING = True    # tile VAE untuk hemat memori

# Setting video (disesuaikan untuk 4GB)
DEFAULT_WIDTH = 320          # resolusi kecil untuk 4GB
DEFAULT_HEIGHT = 240
DEFAULT_NUM_FRAMES = 25      # ~1 detik di 24fps
DEFAULT_FPS = 24
DEFAULT_STEPS = 40           # Ditingkatkan dari 20 ke 40 untuk mengurangi noise

# Server
HOST = "0.0.0.0"
PORT = 8000
MAX_QUEUE_SIZE = 3
