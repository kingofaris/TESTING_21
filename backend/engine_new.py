# backend/model_handler.py
import torch
import gc
import os
import uuid
from pathlib import Path
from datetime import datetime
from PIL import Image

# PERUBAHAN KRUSIAL: Menggunakan LTXImageToVideoPipeline
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from config import *

class VideoGeneratorModel:
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        if self.is_loaded: return True

        print("[Model] Loading LTX Image-to-Video Pipeline...")
        self._clear_vram()

        try:
            # Gunakan class ImageToVideo
            self.pipeline = LTXImageToVideoPipeline.from_pretrained(
                MODEL_ID, # Mengambil dari jalur lokal D: Anda
                torch_dtype=torch.bfloat16,
                cache_dir=str(MODEL_CACHE_DIR),
                local_files_only=False, # Memaksa mode offline
                low_cpu_mem_usage=True,
            )

            # Optimasi VRAM 6GB
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()

            self.is_loaded = True
            print(f"[Model] ✅ Model I2V siap! Sisa VRAM: {self._get_free_vram():.1f} GB")
            return True
        except Exception as e:
            print(f"[Model] ❌ Gagal load model: {e}")
            self.pipeline = None
            self.is_loaded = False
            return False

    def generate_video(
        self,
        image_path: str, # Menerima gambar lokal
        prompt: str,
        negative_prompt: str,
        width: int = 512,
        height: int = 320,
        num_frames: int = 33,
        num_inference_steps: int = 35,
        guidance_scale: float = 3.0,
        seed: int = -1,
    ) -> dict:

        if not self.is_loaded:
            if not self.load_model():
                return {"success": False, "error": "Gagal meload model."}

        # Dimensi wajib kelipatan 32 untuk LTX
        width = (width // 32) * 32
        height = (height // 32) * 32
        
        # 1. Siapkan Gambar Referensi
        try:
            # Buka gambar, ubah ke format RGB, dan paksa resolusinya sesuai output
            # Ini sangat penting agar VRAM tidak OOM karena perbedaan ukuran gambar input
            init_image = load_image(image_path).convert("RGB")
            init_image = init_image.resize((width, height), Image.LANCZOS)
        except Exception as e:
            return {"success": False, "error": f"Gagal membaca gambar: {e}"}

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
            
        # Set global seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            self._clear_vram()

            with torch.inference_mode():
                # Masukkan gambar ke pipeline
                output = self.pipeline(
                    image=init_image, 
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"i2v_{timestamp}_{str(uuid.uuid4())[:8]}.mp4"
            output_path = OUTPUT_DIR / filename

            import imageio
            import numpy as np
            video_frames = [np.array(frame) for frame in output.frames[0]]
            imageio.mimsave(str(output_path), video_frames, fps=8, codec="libx264")
            self._clear_vram()

            return {
                "success": True,
                "video_path": str(output_path),
                "filename": filename,
                "seed": seed,
            }

        except torch.cuda.OutOfMemoryError:
            self._clear_vram()
            return {"success": False, "error": "OOM (Kehabisan VRAM). Turunkan jumlah frame."}
        except Exception as e:
            self._clear_vram()
            return {"success": False, "error": str(e)}

video_generator = VideoGeneratorModel()