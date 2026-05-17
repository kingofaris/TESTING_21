@echo off
title Setup & Install Dependencies
cd /d "%~dp0"
color 0E

echo.
echo  ============================================================
echo   SETUP - Install Semua Dependencies
echo   RTX 4050 6GB - LTX Video Project
echo  ============================================================
echo.

:: Step 1: Buat venv
echo  [1/4] Membuat virtual environment...
if exist "venv\Scripts\python.exe" (
    echo         venv sudah ada, skip.
) else (
    python -m venv venv
    echo         venv berhasil dibuat!
)
echo.

:: Aktifkan
call venv\Scripts\activate.bat

:: Step 2: Install PyTorch CUDA
echo  [2/4] Install PyTorch dengan CUDA 12.1...
echo        (File ~2.5GB, butuh waktu...)
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.

:: Step 3: Install FastAPI & dependencies
echo  [3/4] Install FastAPI dan dependencies lain...
pip install fastapi==0.115.0 uvicorn==0.30.0 python-multipart aiofiles python-dotenv
pip install diffusers transformers accelerate sentencepiece protobuf opencv-python
pip install imageio imageio-ffmpeg
pip install huggingface_hub hf_transfer

echo.

:: Step 4: Cek GPU
echo  [4/4] Verifikasi GPU...
python test_gpu.py

echo.
echo  ============================================================
echo   SETUP SELESAI!
echo   
echo   Langkah berikutnya:
echo   1. Download model: double-click download_model.bat
echo   2. Jalankan server: double-click start_server.bat
echo  ============================================================
echo.
pause
