@echo off
title LTX Video - AI Text-to-Video Server
cd /d "%~dp0"
color 0A

echo.
echo  ============================================================
echo   AI Text-to-Video Generator - Startup Script
echo   LTX Video + FastAPI + RTX 4050 46B
echo  ============================================================
echo.

:: Cek apakah virtual environment ada
if not exist "venv\Scripts\python.exe" (
    echo  [ERROR] Virtual environment tidak ditemukan!
    echo  Buat dulu dengan menjalankan: setup.bat
    echo.
    pause
    exit /b 1
)

:: Aktifkan venv
echo  [1/3] Mengaktifkan virtual environment...
call venv\Scripts\activate.bat

:: Set VRAM optimization untuk 4GB
echo  [2/3] Setting optimasi VRAM 4GB...
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set HF_XET_HIGH_PERFORMANCE=1

:: Jalankan server                                                                                          
echo  [3/3] Menjalankan FastAPI server...
echo.
echo  ============================================================
echo   Server  : http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Frontend: Buka frontend\index.html di browser
echo  ============================================================
echo.

cd backend
python main.py

pause
