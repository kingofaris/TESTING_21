# AI Text-to-Video Generator (LTX-Video)

Sistem ini adalah **aplikasi web Text-to-Video berbasis AI** yang dirancang secara khusus untuk dapat dijalankan secara lokal pada PC/Laptop dengan spesifikasi **VRAM terbatas (seperti Nvidia RTX 3050 4GB)**.

Aplikasi ini menggunakan model **LTX-Video (Lightricks 2B)** yang berjalan di atas framework **FastAPI** (Backend) dan **Vanilla HTML/CSS/JS** (Frontend).

## Fitur Utama
- 🚀 **Optimasi VRAM (4GB)**: Menggunakan CPU Offloading, VAE Slicing, VAE Tiling, dan tipe data `bfloat16` agar model dapat berjalan tanpa Out of Memory (OOM).
- 🎬 **Text-to-Video**: Mampu menghasilkan video berdurasi singkat (~1 detik / 25 frame) pada resolusi 320x240 dengan inferensi cepat.
- 💻 **Web Dashboard**: Antarmuka pengguna (UI) lokal yang intuitif untuk memasukkan prompt dan memonitor status generasi video secara real-time.
- 🛠️ **Server Lokal**: Berjalan 100% offline secara lokal setelah model berhasil diunduh.
- 📦 **Automated Scripts**: Dilengkapi dengan berbagai script `.bat` siap pakai untuk Windows (`setup.bat`, `start_server.bat`, `download_model.bat`).

## Struktur Proyek
```text
📦 model baru
 ┣ 📂 backend/         # Logika server FastAPI & Text-to-Video pipeline
 ┣ 📂 frontend/        # Web dashboard (UI)
 ┣ 📂 models/          # Tempat cache untuk model HuggingFace
 ┣ 📂 outputs/         # Hasil video yang dihasilkan (.mp4)
 ┣ 📂 venv/            # Python virtual environment (terbuat saat instalasi)
 ┣ 📜 .env             # Konfigurasi variabel lingkungan lokal
 ┣ 📜 INSTALL_GUIDE.txt# Instruksi instalasi manual lengkap
 ┣ 📜 download_model.bat # Script untuk mendownload model manual (jika butuh)
 ┣ 📜 requirements.txt # Daftar dependensi Python
 ┣ 📜 setup.bat        # Script untuk setup awal otomatis
 ┣ 📜 start_server.bat # Script untuk menyalakan web server & frontend
 ┗ 📜 test_gpu.py      # Script untuk cek penggunaan dan deteksi GPU
```

## Prasyarat
Sistem ini membutuhkan sistem operasi **Windows** dan disarankan memiliki spesifikasi berikut:
- **GPU**: NVIDIA GPU dengan minimal **4GB VRAM** (Contoh: RTX 3050 Laptop).
- **RAM**: Minimal 16GB.
- **Python**: Versi 3.10 atau 3.11.

## Cara Instalasi

### Menggunakan Script Otomatis (Termudah)
1. Buka folder proyek ini.
2. Double klik file `setup.bat`.
3. Tunggu hingga proses instalasi virtual environment dan library Pytorch beserta CUDA selesai diunduh.

### Instalasi Manual (Jika setup.bat gagal)
Jika terdapat masalah, silakan mengacu ke dokumen `INSTALL_GUIDE.txt` untuk langkah demi langkah penginstalan manual menggunakan terminal. Instruksi mencakup instalasi PyTorch (CUDA 12.1) dan dependencies.

## Cara Menggunakan
1. Jalankan aplikasi dengan melakukan klik ganda pada `start_server.bat`. 
   > Pada pemanggilan pertama setelah instalasi, proses ini akan mendownload model **LTX-Video** ke dalam folder `models/` (bisa memakan waktu beberapa menit tergantunng kecepatan internet).
2. Command prompt akan berjalan dan mengaktifkan virtual environment serta Server FastAPI.
3. Buka browser dan jalankan aplikasinya pada:
   - **Frontend UI / Website**: Anda bisa langsung membuka file `frontend\index.html` dari File Explorer.
   - **API Docs (Swagger)**: `http://localhost:8000/docs`
4. Masukkan prompt / instruksi teks pada antarmuka web, klik tombol Generate, dan tunggu hasil MP4 muncul. Hasil asli video akan tersimpan ke folder `outputs/`.

## Troubleshooting & Tips VRAM 4GB
Jika Anda mengalami error **CUDA Out of Memory (OOM)**:
- Kurangi resolusi pada `backend/config.py` dari `320x240` ke yang lebih rendah.
- Kurangi opsi jumlah frame `DEFAULT_NUM_FRAMES` jadi rentang `8-16` saja.
- Pastikan tidak ada aplikasi berat lain (Game, Rendering, Browser dengan banyak tab) yang menyita VRAM ketika menekan "Generate".
- Model bitandbytes terkadang bermasalah pada OS Windows murni. Jika library gagal di-load, baca file `INSTALL_GUIDE.txt` tentang cara reinstall bitsandbytes versi windows.
