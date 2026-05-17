# backend/model_handler.py
import torch
import gc
import os
import uuid
from pathlib import Path
from datetime import datetime
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from config import *

class VideoGeneratorModel:
    """
    Handler untuk LTX Video dioptimalkan untuk RTX 4050 6GB VRAM dan RAM sistem ~10GB Free.
    Fokus pada peningkatan kualitas render (Inference Steps & CFG).
    """

    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        print(f"[Model] Device: {self.device}")

    def _get_free_vram(self):
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1024**3
        return 0

    def _clear_vram(self):
        """Pembersihan cache agresif untuk menghindari OOM saat transisi proses."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        if self.is_loaded:
            return True

        print("[Model] Loading LTX Video Pipeline...")
        self._clear_vram()

        try:
            # Menggunakan bfloat16 untuk kualitas warna yang lebih baik dan hemat VRAM
            self.pipeline = LTXPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=str(MODEL_CACHE_DIR),
                low_cpu_mem_usage=True, # Mencegah RAM sistem melonjak terlalu tinggi
            )

            # ===== OPTIMASI UNTUK 6GB VRAM =====
            
            # Offloading krusial: Memindahkan komponen yang tidak aktif ke RAM sistem.
            # Karena Anda memiliki ~10GB free RAM, fitur ini aman dinyalakan.
            self.pipeline.enable_model_cpu_offload()

            # Decoding per-frame (Slicing) sangat penting untuk VRAM menengah
            self.pipeline.vae.enable_slicing()

            # Mencegah OOM saat resolusi ditarik agak tinggi
            self.pipeline.vae.enable_tiling()

            # ====================================

            self.is_loaded = True
            print(f"[Model] ✅ Model siap! Sisa VRAM: {self._get_free_vram():.1f} GB")
            return True

        except Exception as e:
            print(f"[Model] ❌ Gagal load model: {e}")
            self.pipeline = None
            self.is_loaded = False
            return False

    def unload_model(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            self._clear_vram()
            print("[Model] Model di-unload dari memori.")

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int = 512,
        height: int = 320,
        num_frames: int = 25,
        num_inference_steps: int = 35, # Default ditingkatkan untuk kualitas yang lebih halus
        guidance_scale: float = 4.0,   # CFG diturunkan sedikit dari default diffusers agar tidak over-fried
        seed: int = -1,
    ) -> dict:
        """Eksekusi Text-to-Video dengan parameter dari UI."""

        if not self.is_loaded:
            success = self.load_model()
            if not success:
                return {"success": False, "error": "Gagal meload model ke memori."}

        # Keamanan dimensi untuk LTX (Harus kelipatan 32)
        width = (width // 32) * 32
        height = (height // 32) * 32
        
        # Keamanan jumlah frame (Biasanya LTX meminta num_frames = 8k + 1, diffusers menangani sebagian)
        num_frames = min(num_frames, 60) 

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator().manual_seed(seed)

        try:
            self._clear_vram()

            with torch.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            # Ekspor Video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = str(uuid.uuid4())[:8]
            filename = f"render_{timestamp}_{video_id}.mp4"
            output_path = OUTPUT_DIR / filename

            # Export dengan 8 FPS atau disesuaikan dengan kebutuhan Anda
            export_to_video(output.frames[0], str(output_path), fps=8)

            self._clear_vram()

            return {
                "success": True,
                "video_path": str(output_path),
                "filename": filename,
                "seed": seed,
            }

        except torch.cuda.OutOfMemoryError:
            self._clear_vram()
            return {"success": False, "error": "VRAM 6GB kehabisan kapasitas. Turunkan jumlah frame atau tutup aplikasi berat lainnya."}
        except Exception as e:
            self._clear_vram()
            return {"success": False, "error": str(e)}

# Inisialisasi Singleton
video_generator = VideoGeneratorModel()