# backend/main.py
import asyncio
import uuid
import time
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from model_handler import video_generator
from config import OUTPUT_DIR, HOST, PORT

app = FastAPI(
    title="Text-to-Video AI API",
    description="Generate video dari teks menggunakan LTX Video",
    version="1.0.0"
)

# CORS - izinkan request dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Ganti dengan domain kamu di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve video output
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve frontend (opsional, kalau ingin satu server)
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ===== MODELS (Request/Response Schema) =====

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500, description="Deskripsi video yang ingin dibuat")
    negative_prompt: str = Field(default="worst quality, inconsistent, blurry, noisy, distorted, artifact, low resolution", description="Hal yang tidak diinginkan")
    width: int = Field(default=320, ge=64, le=512)
    height: int = Field(default=256, ge=64, le=512)
    num_frames: int = Field(default=25, ge=8, le=1500, description="Jumlah frame (8-1500)")
    num_inference_steps: int = Field(default=40, ge=4, le=100)
    guidance_scale: float = Field(default=3.5, ge=1.0, le=10.0)
    seed: int = Field(default=-1, description="Seed (-1 = random)")


class GenerateResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    video_url: Optional[str] = None
    filename: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


# Queue sederhana untuk antrian generate
generation_queue = asyncio.Queue(maxsize=3)
job_status = {}  # {"job_id": {"status": "...", "result": {...}}}


# ===== ENDPOINTS =====

@app.get("/")
def root():
    return {"message": "Text-to-Video API aktif!", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Cek status server dan model."""
    import torch
    gpu_info = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total / 1024**3, 1),
            "vram_free_gb": round(free / 1024**3, 1),
        }
    # Cari apakah ada job yang sedang processing
    active_job = None
    for jid, info in job_status.items():
        if info["status"] == "processing":
            active_job = jid
            break

    return {
        "status": "ok",
        "model_loaded": video_generator.is_loaded,
        "gpu": gpu_info,
        "queue_size": generation_queue.qsize(),
        "active_job": active_job,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate video dari text prompt.
    Karena generate bisa lambat, ini pakai background task + job ID.
    """
    if generation_queue.full():
        raise HTTPException(
            status_code=429,
            detail="Antrian penuh (max 3). Coba lagi dalam beberapa menit."
        )

    job_id = str(uuid.uuid4())[:12]
    job_status[job_id] = {"status": "queued", "result": None}

    # Jalankan generate di background
    background_tasks.add_task(
        _run_generation,
        job_id=job_id,
        request=request
    )

    return GenerateResponse(
        success=True,
        job_id=job_id,
    )


async def _run_generation(job_id: str, request: GenerateRequest):
    """Background task untuk generate video."""
    job_status[job_id]["status"] = "processing"
    start = time.time()

    try:
        # Jalankan di thread pool agar tidak block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # ThreadPoolExecutor default
            lambda: video_generator.generate_video(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )
        )

        duration = round(time.time() - start, 1)

        if result["success"]:
            video_url = f"/outputs/{result['filename']}"
            job_status[job_id] = {
                "status": "done",
                "result": {
                    "success": True,
                    "video_url": video_url,
                    "filename": result["filename"],
                    "seed": result["seed"],
                    "duration_seconds": duration,
                }
            }
        else:
            job_status[job_id] = {
                "status": "failed",
                "result": {"success": False, "error": result.get("error", "Unknown error")}
            }

    except Exception as e:
        job_status[job_id] = {
            "status": "failed",
            "result": {"success": False, "error": str(e)}
        }


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Cek status job generate video."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan")

    job = job_status[job_id]
    response = {
        "job_id": job_id, 
        "status": job["status"],
        "model_loaded": video_generator.is_loaded
    }

    if job["status"] in ["done", "failed"] and job["result"]:
        response.update(job["result"])

    return response


@app.get("/videos")
def list_videos():
    """Daftar semua video yang sudah digenerate."""
    import json
    videos = []
    for f in sorted(OUTPUT_DIR.glob("*.mp4"), reverse=True):
        stat = f.stat()
        video_data = {
            "filename": f.name,
            "url": f"/outputs/{f.name}",
            "size_mb": round(stat.st_size / 1024**2, 2),
            "created_at": stat.st_mtime,
        }
        
        json_file = f.with_suffix('.json')
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as jf:
                    meta = json.load(jf)
                    video_data["seed"] = meta.get("seed")
                    video_data["prompt"] = meta.get("prompt")
                    video_data["duration_seconds"] = meta.get("duration_seconds")
            except Exception:
                pass
                
        videos.append(video_data)
    return {"videos": videos[:20]}  # tampilkan 20 terbaru


@app.delete("/videos/{filename}")
def delete_video(filename: str):
    """Hapus video tertentu."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    filepath.unlink()
    
    json_path = filepath.with_suffix('.json')
    if json_path.exists():
        json_path.unlink()
        
    return {"message": f"{filename} berhasil dihapus"}


@app.post("/model/load")
def load_model():
    """Load model AI secara manual."""
    success = video_generator.load_model()
    return {"success": success, "model_loaded": video_generator.is_loaded}


@app.post("/model/unload")
def unload_model():
    """Unload model dari memori."""
    video_generator.unload_model()
    return {"success": True, "model_loaded": video_generator.is_loaded}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,   # Jangan pakai reload=True, berat untuk GPU
        workers=1,      # WAJIB 1 worker saja untuk GPU sharing
    )
