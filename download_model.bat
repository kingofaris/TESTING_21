@echo off
title LTX Video - Model Downloader
cd /d "%~dp0"
color 0B

echo.
echo  ============================================================
echo   LTX Video Model Downloader
echo   Download otomatis dengan RESUME support
echo  ============================================================
echo.

:: Cek venv
if not exist "venv\Scripts\python.exe" (
    echo  [SETUP] Membuat virtual environment...
    python -m venv venv
)

:: Aktifkan venv
call venv\Scripts\activate.bat

:: Install huggingface_hub & hf_transfer dulu
echo  [Setup] Memastikan tools download terinstall...
pip install huggingface_hub hf_transfer -q

:: Set download acceleration
set HF_HUB_ENABLE_HF_TRANSFER=1

echo.
echo  [Info] Model yang akan didownload: Lightricks/LTX-Video
echo  [Info] Estimasi ukuran: ~5-6 GB
echo  [Info] Download bisa dilanjutkan jika terputus! (resume otomatis)
echo.

:: Jalankan download script
python download_model.py

echo.
pause
