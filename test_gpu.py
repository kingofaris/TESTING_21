# test_gpu.py
# Jalankan dengan: python test_gpu.py
import torch

print("=" * 40)
print("GPU & CUDA Check")
print("=" * 40)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    print(f"VRAM Total      : {round(total / 1024**3, 1)} GB")
    print(f"VRAM Bebas      : {round(free / 1024**3, 1)} GB")
    print(f"CUDA version    : {torch.version.cuda}")
    print("\n✅ GPU siap digunakan!")
else:
    print("\n⚠️  CUDA tidak tersedia. Cek instalasi PyTorch + CUDA.")
    print("    Jalankan: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
