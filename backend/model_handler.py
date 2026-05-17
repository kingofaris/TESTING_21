# backend/model_handler.py
import torch
import gc
import os
import uuid
from pathlib import Path
from datetime import datetime
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from config import *


class VideoGeneratorModel:
    """
    Handler untuk LTX Video dengan optimasi VRAM 4GB.
    Menggunakan CPU offloading agar bisa jalan di GPU terbatas.
    """

    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        print(f"[Model] Device: {self.device}")
        print(f"[Model] VRAM tersedia: {self._get_free_vram():.1f} GB")

    def _get_free_vram(self):
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1024**3
        return 0

    def _clear_vram(self):
        """Bersihkan VRAM sebelum load model."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        """Load LTX Video pipeline dengan optimasi 4GB VRAM."""
        if self.is_loaded:
            print("[Model] Model sudah ter-load.")
            return True

        print("[Model] Loading LTX Video... (bisa 2-5 menit pertama kali)")
        self._clear_vram()

        try:
            # Load pipeline dengan dtype hemat memori
            self.pipeline = LTXPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=str(MODEL_CACHE_DIR),
                low_cpu_mem_usage=True,
            )

            # ===== OPTIMASI WAJIB UNTUK 4GB VRAM =====

            # 1. CPU Offloading: model pindah ke CPU saat tidak dipakai
            #    Ini yang paling penting untuk 4GB!
            self.pipeline.enable_model_cpu_offload()

            # 2. VAE Slicing: decode video per-frame, bukan sekaligus
            self.pipeline.vae.enable_slicing()

            # 3. VAE Tiling: decode per-tile untuk resolusi besar
            self.pipeline.vae.enable_tiling()

            # 4. Attention slicing: potong attention jadi bagian kecil
            self.pipeline.enable_attention_slicing(slice_size="auto")

            # ==========================================

            self.is_loaded = True
            print(f"[Model] ✅ Model berhasil di-load!")
            print(f"[Model] VRAM sisa: {self._get_free_vram():.1f} GB")
            return True

        except Exception as e:
            print(f"[Model] ❌ Gagal load model: {e}")
            self.pipeline = None
            self.is_loaded = False
            return False

    def unload_model(self):
        """Unload model dari memori (opsional, untuk hemat RAM)."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            self._clear_vram()
            print("[Model] Model di-unload dari memori.")

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, blurry, low resolution",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = 3.0,
        seed: int = -1,
    ) -> dict:
        """
        Generate video dari text prompt.

        Returns:
            dict: {"success": bool, "video_path": str, "error": str}
        """

        if not self.is_loaded:
            success = self.load_model()
            if not success:
                return {"success": False, "error": "Gagal load model AI"}

        # Validasi input untuk 4GB VRAM
        # Batasi resolusi dan frame agar tidak OOM
        width = min(width, 512)
        height = min(height, 512)
        num_frames = min(num_frames, 1500)  # max limit updated for longer videos

        # Set seed untuk reproducibility
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
            
        # Set global seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Opsional: Uncomment baris di bawah ini jika ingin deterministik penuh
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[Generate] Prompt: {prompt[:50]}...")
        print(f"[Generate] Resolusi: {width}x{height}, Frames: {num_frames}")
        print(f"[Generate] Seed: {seed}")

        try:
            self._clear_vram()

            # Generate video
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

            # Simpan video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = str(uuid.uuid4())[:8]
            filename = f"video_{timestamp}_{video_id}.mp4"
            output_path = OUTPUT_DIR / filename

            # pyrefly: ignore [missing-import]
            import imageio
            import numpy as np
            
            video_frames = [np.array(frame) for frame in output.frames[0]]
            imageio.mimsave(
                str(output_path),
                video_frames,
                fps=DEFAULT_FPS,
                codec="libx264"
            )

            import json
            metadata = {
                "prompt": prompt,
                "seed": seed,
                "duration_seconds": round(num_frames / DEFAULT_FPS, 1)
            }
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            print(f"[Generate] ✅ Video tersimpan: {filename}")
            self._clear_vram()

            return {
                "success": True,
                "video_path": str(output_path),
                "filename": filename,
                "seed": seed,
                "width": width,
                "height": height,
                "num_frames": num_frames,
            }

        except torch.cuda.OutOfMemoryError:
            self._clear_vram()
            return {
                "success": False,
                "error": "VRAM habis (Out of Memory). Coba kurangi resolusi atau jumlah frame."
            }
        except Exception as e:
            self._clear_vram()
            return {"success": False, "error": str(e)}


# Singleton - satu instance untuk seluruh aplikasi
video_generator = VideoGeneratorModel()
