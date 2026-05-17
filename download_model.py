"""
============================================================
  DOWNLOAD SCRIPT - LTX Video Model
  Optimasi untuk RTX 3050 4GB VRAM
  
  Cara pakai:
    python download_model.py
    
  Fitur:
    ✅ Resume download otomatis (tidak ulang dari awal)
    ✅ Progress bar real-time
    ✅ Download hanya file yang diperlukan saja
    ✅ Verifikasi file setelah download
    ✅ Symlink cache → models/ folder project
============================================================
"""

import os
import sys
import time
from pathlib import Path

# ── Pastikan hf_transfer & huggingface_hub terinstall ──────
def install_pkg(pkg):
    import subprocess
    print(f"[Setup] Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub import login as hf_login
except ImportError:
    install_pkg("huggingface_hub")
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub import login as hf_login

try:
    import hf_transfer  # noqa
    # Seringkali hf_transfer gagal tanpa token atau di env tertentu. 
    # Kita matikan dulu secara default untuk stabilitas.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    HF_TRANSFER_AVAILABLE = False

# ============================================================
#  KONFIGURASI
# ============================================================
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model ID di HuggingFace
# LTX-Video 2B — model resmi Lightricks
MODEL_ID = "Lightricks/LTX-Video"

# File yang WAJIB didownload (target v0.9.5 agar hemat storage)
# Kita hanya ambil file pendukung dan SATU weight model terbaru
REQUIRED_PATTERNS = [
    "config.json",
    "*.json",
    "*.txt",
    "*tokenizer*",
    "*vae*",
    "*transformer*",
    "ltx-video-2b-v0.9.5.safetensors", # Hanya ambil versi terbaru
]

# File yang benar-benar dilarang (versi lama)
IGNORE_PATTERNS = [
    "ltx-video-2b-v0.9.safetensors",
    "ltx-video-2b-v0.9.1.safetensors",
    "ltxv-13b*",
    "*.bin",
    "*.msgpack",
]

# ============================================================
#  CEK & PASANG HF TOKEN (Opsional tapi disarankan)
# ============================================================
def setup_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    
    # Coba baca dari .env file
    env_file = BASE_DIR / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print(f"[Auth] OK HF Token ditemukan (***{token[-4:]})")
        return True
    else:
        print("[Auth] !! HF Token tidak ditemukan.")
        print("       Model LTX-Video bersifat publik, download tetap bisa jalan.")
        print("       Untuk kecepatan lebih baik, tambah token di file .env:")
        print("       HF_TOKEN=hf_xxxxxx")
        print()
        return False

# ============================================================
#  INSTALL hf_transfer UNTUK DOWNLOAD 3-5x LEBIH CEPAT
# ============================================================
def try_install_hf_transfer():
    global HF_TRANSFER_AVAILABLE
    if HF_TRANSFER_AVAILABLE:
        print("[Speed] OK hf_transfer terdeteksi -- (Dinonaktifkan demi stabilitas)")
        return
    
    print("[Speed] Mencoba install hf_transfer untuk download lebih cepat...")
    try:
        install_pkg("hf_transfer")
        import hf_transfer  # noqa
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        HF_TRANSFER_AVAILABLE = True
        print("[Speed] OK hf_transfer berhasil diinstall!")
    except Exception:
        print("[Speed] !! hf_transfer gagal (tidak masalah, download tetap jalan)")

# ============================================================
#  CEK APAKAH MODEL SUDAH ADA
# ============================================================
def check_existing_model():
    """Cek apakah model sudah pernah didownload."""
    local_model_path = MODEL_DIR / MODEL_ID.replace("/", "--")
    
    # Cek di HuggingFace cache default juga
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    cache_model_path = hf_cache / f"models--{MODEL_ID.replace('/', '--')}"
    
    if local_model_path.exists():
        safetensors = list(local_model_path.rglob("*.safetensors"))
        if safetensors:
            total_size = sum(f.stat().st_size for f in safetensors) / 1024**3
            print(f"[Check] OK Model ditemukan di: {local_model_path}")
            print(f"[Check]    File .safetensors: {len(safetensors)} file ({total_size:.1f} GB)")
            return True, str(local_model_path)
    
    if cache_model_path.exists():
        safetensors = list(cache_model_path.rglob("*.safetensors"))
        if safetensors:
            total_size = sum(f.stat().st_size for f in safetensors) / 1024**3
            print(f"[Check] OK Model ditemukan di HF cache: {cache_model_path}")
            print(f"[Check]    File .safetensors: {len(safetensors)} file ({total_size:.1f} GB)")
            return True, str(cache_model_path)
    
    print(f"[Check] Model belum ada, akan didownload...")
    return False, None

# ============================================================
#  MAIN DOWNLOAD
# ============================================================
def download_model():
    print()
    print("=" * 60)
    print("  LTX Video Model Downloader")
    print(f"  Model: {MODEL_ID}")
    print(f"  Simpan ke: {MODEL_DIR}")
    print("=" * 60)
    print()
    
    # Setup
    setup_hf_token()
    try_install_hf_transfer()
    
    print()
    
    # Cek model sudah ada
    exists, existing_path = check_existing_model()
    if exists:
        print()
        user_input = input("Model sudah ada. Download ulang? (y/N): ").strip().lower()
        if user_input != "y":
            print("\nOK Skip download. Model siap digunakan!")
            print_usage_info(existing_path)
            return existing_path
    
    print()
    print("[Download] Mulai download model LTX Video...")
    print("[Download] Estimasi size: ~5-6 GB")
    print("[Download] Progress akan muncul di bawah ini...")
    print()
    
    # Set cache dir ke folder models/ project kita
    local_dir = str(MODEL_DIR / MODEL_ID.replace("/", "--"))
    
    start_time = time.time()
    
    try:
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=local_dir,           # simpan di folder project
            local_dir_use_symlinks=False,   # copy file asli (tidak pakai symlink)
            allow_patterns=REQUIRED_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            resume_download=True,           # RESUME otomatis jika terputus!
            max_workers=4,                  # 4 koneksi paralel
        )
        
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        print()
        print("=" * 60)
        print(f"  OK DOWNLOAD SELESAI! ({mins}m {secs}s)")
        print(f"  Path: {local_path}")
        print("=" * 60)
        
        # Verifikasi
        verify_download(local_path)
        print_usage_info(local_path)
        
        return local_path
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("  PAUSE Download dihentikan manual.")
        print("  Jalankan script ini lagi untuk MELANJUTKAN download")
        print("  (tidak perlu mulai dari awal!)")
        print("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        print()
        print(f"[Error] !! Download gagal: {e}")
        print()
        print("Kemungkinan penyebab:")
        print("  1. Koneksi internet terputus -> jalankan script lagi")
        print("  2. Storage penuh -> pastikan ada 10GB kosong")
        print("  3. Rate limit HuggingFace -> tunggu 5 menit, coba lagi")
        print()
        print("Script bisa dilanjutkan tanpa kehilangan progress! OK")
        sys.exit(1)

# ============================================================
#  VERIFIKASI FILE
# ============================================================
def verify_download(local_path: str):
    path = Path(local_path)
    
    # Cek file penting
    safetensors = list(path.rglob("*.safetensors"))
    json_files  = list(path.rglob("*.json"))
    
    print()
    print("[Verify] Memeriksa file download...")
    
    if not safetensors:
        print("[Verify] !! Tidak ada file .safetensors! Download mungkin gagal.")
        return False
    
    total_size_gb = sum(f.stat().st_size for f in safetensors) / 1024**3
    
    print(f"[Verify] OK .safetensors : {len(safetensors)} file ({total_size_gb:.2f} GB)")
    print(f"[Verify] OK .json config : {len(json_files)} file")
    
    # List semua safetensors
    print()
    print("[Verify] File model:")
    for f in sorted(safetensors):
        size_gb = f.stat().st_size / 1024**3
        print(f"         [File] {f.name}  ({size_gb:.2f} GB)")
    
    return True

# ============================================================
#  INFO SETELAH DOWNLOAD
# ============================================================
def print_usage_info(local_path: str):
    print()
    print("=" * 60)
    print("  LANGKAH SELANJUTNYA")
    print("=" * 60)
    print()
    print("  1. Update config.py jika perlu menggunakan path lokal:")
    print(f"     MODEL_CACHE_DIR = Path(r'{local_path}')")
    print()
    print("  2. Jalankan backend server:")
    print("     → Double-click: start_server.bat")
    print("     → Atau manual:  cd backend && python main.py")
    print()
    print("  3. Buka frontend di browser:")
    print("     → Buka file: frontend/index.html")
    print()
    print("  ⚡ Tips untuk RTX 3050 4GB:")
    print("     • Resolusi: 320x240 atau 480x272")
    print("     • Frames  : 16-25 frame")
    print("     • Steps   : 15-20 (cukup untuk kualitas baik)")
    print()

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Cek Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ diperlukan!")
        sys.exit(1)
    
    download_model()
