# backend/main.py
import asyncio
import uuid
import time
import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from engine_new import video_generator
from config import OUTPUT_DIR, HOST, PORT

app = FastAPI(title="LTX Image-to-Video API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


generation_queue = asyncio.Queue(maxsize=3)
job_status = {}  

@app.get("/health")
def health_check():
    import torch
    gpu_info = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_info = {"vram_free_gb": round(free / 1024**3, 1)}
    return {"status": "ok", "gpu": gpu_info}


# Endpoint diubah untuk menerima File dan Form Data
@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("worst quality, inconsistent, blurry, noisy, distorted, artifact, low resolution"),
    width: int = Form(512),
    height: int = Form(320),
    num_frames: int = Form(33),
    num_inference_steps: int = Form(40),
    guidance_scale: float = Form(3.5),
    seed: int = Form(-1)
):
    if generation_queue.full():
        raise HTTPException(status_code=429, detail="Antrian penuh.")

    # 1. Simpan gambar sementara ke disk
    temp_img_name = f"temp_{uuid.uuid4().hex[:8]}.jpg"
    temp_img_path = OUTPUT_DIR / temp_img_name
    with open(temp_img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    job_id = str(uuid.uuid4())[:12]
    job_status[job_id] = {"status": "queued", "result": None}

    # 2. Kirim data ke background task
    background_tasks.add_task(
        _run_generation,
        job_id=job_id,
        image_path=str(temp_img_path),
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed
    )

    return {"success": True, "job_id": job_id}


async def _run_generation(job_id, image_path, prompt, negative_prompt, width, height, num_frames, num_inference_steps, guidance_scale, seed):
    job_status[job_id]["status"] = "processing"
    start = time.time()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  
            lambda: video_generator.generate_video(
                image_path=image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        )

        duration = round(time.time() - start, 1)
        
        # Hapus gambar sementara setelah selesai
        if os.path.exists(image_path):
            os.remove(image_path)

        if result["success"]:
            job_status[job_id] = {
                "status": "done",
                "result": {
                    "success": True,
                    "video_url": f"/outputs/{result['filename']}",
                    "filename": result["filename"],
                    "seed": result["seed"],
                    "duration_seconds": duration,
                }
            }
        else:
            job_status[job_id] = {"status": "failed", "result": {"success": False, "error": result.get("error", "")}}

    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        job_status[job_id] = {"status": "failed", "result": {"success": False, "error": str(e)}}


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan")
    job = job_status[job_id]
    response = {"job_id": job_id, "status": job["status"]}
    if job["status"] in ["done", "failed"] and job["result"]:
        response.update(job["result"])
    return response


if __name__ == "__main__":
    uvicorn.run("new_main:app", host=HOST, port=PORT, reload=False, workers=1)