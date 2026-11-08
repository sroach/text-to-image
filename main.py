import asyncio
import io
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

# Base directory — absolute path so it works regardless of working directory
BASE_DIR = Path(__file__).resolve().parent

# Global pipeline and lock
pipeline = None
generation_lock = asyncio.Lock()

# Directory for saved outputs
OUTPUT_DIR = BASE_DIR / "generated_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def get_optimal_device_and_dtype():
    """Detect optimal compute backend and precision, tailored for Apple Silicon (MPS)."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        # Apple Silicon Mac GPU (Metal Performance Shaders)
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def sanitize_filename(filename: str) -> str:
    """Ensure safe filename with .png extension."""
    if not filename or not filename.strip():
        filename = "generated_image.png"
    # Extract only base filename to prevent directory traversal
    filename = Path(filename).name
    # Remove unsafe characters
    filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    return filename


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once into memory on startup."""
    global pipeline
    device, dtype = get_optimal_device_and_dtype()
    print(f"🚀 Loading diffusion model on device: {device.upper()} with dtype: {dtype}...")

    # On Apple Silicon (MPS), replace the default SDXL VAE with a fixed version.
    # The default SDXL VAE produces NaN/Inf values on MPS, resulting in black images.
    # The community-patched VAE (madebyollin/sdxl-vae-fp16-fix) works correctly in float16 on MPS.
    vae = None
    if device == "mps":
        print("🔄 Loading MPS-compatible VAE (madebyollin/sdxl-vae-fp16-fix)...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=dtype,
        )

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        vae=vae,
        use_safetensors=True,
    )

    # Move entire pipeline to device - all components share the same dtype
    pipeline.to(device)

    print("✅ Model loaded and ready for inference!")
    yield
    pipeline = None


app = FastAPI(
    title="Text-to-Image Web App",
    description="Generate high quality images from text prompts using SDXL with real-time SSE streaming.",
    lifespan=lifespan,
    root_path=os.environ.get("ROOT_PATH", ""),
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the desired image")
    negative_prompt: Optional[str] = Field(None, description="Things to exclude from the image")
    filename: Optional[str] = Field("output.png", description="Filename for the saved PNG")
    num_inference_steps: int = Field(30, ge=1, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0, description="Classifier-free guidance scale")
    width: Optional[int] = Field(None, ge=256, le=2048, description="Image width (SDXL default: 1024)")
    height: Optional[int] = Field(None, ge=256, le=2048, description="Image height (SDXL default: 1024)")


def _run_inference(
    prompt: str, steps: int, guidance: float, save_path: Path,
    negative_prompt: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None,
) -> bytes:
    """Synchronous inference function to run in worker thread for direct responses."""
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    result = pipeline(**kwargs).images[0]

    # Save to local disk
    result.save(save_path, format="PNG")

    # Save to byte buffer for HTTP response
    img_byte_arr = io.BytesIO()
    result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def _run_inference_stream(
    prompt: str,
    steps: int,
    guidance: float,
    save_path: Path,
    loop: asyncio.AbstractEventLoop,
    progress_queue: asyncio.Queue,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Path:
    """Synchronous inference with step callback feeding progress into an async queue."""
    def step_callback(pipe, step_index, timestep, callback_kwargs):
        progress = {
            "status": "generating",
            "step": step_index + 1,
            "total_steps": steps,
            "percentage": round(((step_index + 1) / steps) * 100, 1),
        }
        loop.call_soon_threadsafe(progress_queue.put_nowait, progress)
        return callback_kwargs

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "callback_on_step_end": step_callback,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    result = pipeline(**kwargs).images[0]

    result.save(save_path, format="PNG")
    return save_path


async def event_generator(
    prompt: str, steps: int, guidance: float, filename: str,
    negative_prompt: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None,
):
    """Async generator yielding Server-Sent Events (SSE) representing generation progress."""
    if pipeline is None:
        yield f"data: {json.dumps({'status': 'error', 'message': 'Model is not loaded or initializing.'})}\n\n"
        return

    safe_filename = sanitize_filename(filename)
    save_path = OUTPUT_DIR / safe_filename

    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue = asyncio.Queue()

    yield f"data: {json.dumps({'status': 'queued', 'message': 'Waiting for GPU/MPS device lock...', 'percentage': 0})}\n\n"

    async with generation_lock:
        yield f"data: {json.dumps({'status': 'starting', 'message': 'Starting inference on device...', 'percentage': 0})}\n\n"

        task = asyncio.create_task(
            asyncio.to_thread(
                _run_inference_stream,
                prompt=prompt,
                steps=steps,
                guidance=guidance,
                save_path=save_path,
                loop=loop,
                progress_queue=progress_queue,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
            )
        )

        while not task.done() or not progress_queue.empty():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                # Pause BEFORE yielding so each SSE event is spaced out when sent to the client.
                # On fast hardware (MPS/CUDA), all denoising steps complete nearly instantly and flood
                # the queue. Without this delay, the browser receives all events in one TCP packet
                # and only paints the final state, making the progress bar jump from 0% to 100%.
                # 200ms per step gives a smooth ~6 second animation for 30 steps.
                if progress.get("status") == "generating":
                    await asyncio.sleep(0.2)
                yield f"data: {json.dumps(progress)}\n\n"
            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)

        try:
            # Signal that denoising is done and the image is being saved
            yield f"data: {json.dumps({'status': 'saving', 'message': 'Saving generated image...', 'percentage': 100})}\n\n"

            await task

            completion = {
                "status": "completed",
                "step": steps,
                "total_steps": steps,
                "percentage": 100.0,
                "filename": safe_filename,
                "image_url": f"images/{safe_filename}",
            }
            yield f"data: {json.dumps(completion)}\n\n"
        except Exception as e:
            error_data = {
                "status": "error",
                "message": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"


@app.get("/")
async def index(request: Request):
    """Web interface with prompt, filename, parameters, and interactive SSE stream preview."""
    device, dtype = get_optimal_device_and_dtype()
    device_label = "Apple Silicon (MPS)" if device == "mps" else ("NVIDIA GPU (CUDA)" if device == "cuda" else "CPU")

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "device_label": device_label,
        },
    )


@app.get("/images/{filename}")
async def get_image(filename: str):
    """Retrieve saved image from local disk."""
    safe_filename = sanitize_filename(filename)
    image_path = OUTPUT_DIR / safe_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=safe_filename,
    )


@app.get("/api/gallery")
async def gallery():
    """List all generated images with metadata for the gallery view."""
    images = []
    for img_path in OUTPUT_DIR.iterdir():
        if img_path.suffix.lower() == ".png":
            stat = img_path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)
            modified_str = modified.strftime("%Y-%m-%d %H:%M")
            images.append({
                "name": img_path.name,
                "url": f"/images/{img_path.name}",
                "modified": modified_str,
                "size": stat.st_size,
            })
    # Sort by modification time, newest first
    images.sort(key=lambda x: x["modified"], reverse=True)
    return images


@app.post("/generate-stream")
async def generate_stream_form(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    filename: str = Form("output.png"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    width: int = Form(0),
    height: int = Form(0),
):
    """Form endpoint returning an SSE stream of generation status and final image URL."""
    return StreamingResponse(
        event_generator(
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance_scale,
            filename=filename,
            negative_prompt=negative_prompt or None,
            width=width if width > 0 else None,
            height=height if height > 0 else None,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/generate-stream")
async def generate_stream_api(request: GenerateRequest):
    """JSON API endpoint returning an SSE stream of generation status and final image URL."""
    return StreamingResponse(
        event_generator(
            prompt=request.prompt,
            steps=request.num_inference_steps,
            guidance=request.guidance_scale,
            filename=request.filename or "output.png",
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate", response_class=Response)
async def generate_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    filename: str = Form("output.png"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    width: int = Form(0),
    height: int = Form(0),
):
    """Accepts prompt, filename, and inference parameters, saves to disk and returns the PNG image."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still initializing or unavailable.")

    safe_filename = sanitize_filename(filename)
    save_path = OUTPUT_DIR / safe_filename

    async with generation_lock:
        image_bytes = await asyncio.to_thread(
            _run_inference,
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance_scale,
            save_path=save_path,
            negative_prompt=negative_prompt or None,
            width=width if width > 0 else None,
            height=height if height > 0 else None,
        )

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{safe_filename}"',
            "X-Saved-Path": str(save_path),
        },
    )


@app.post("/api/generate", response_class=Response)
async def api_generate_image(request: GenerateRequest):
    """JSON API endpoint accepting prompt, filename, and parameters."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still initializing or unavailable.")

    safe_filename = sanitize_filename(request.filename or "output.png")
    save_path = OUTPUT_DIR / safe_filename

    async with generation_lock:
        image_bytes = await asyncio.to_thread(
            _run_inference,
            prompt=request.prompt,
            steps=request.num_inference_steps,
            guidance=request.guidance_scale,
            save_path=save_path,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
        )

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{safe_filename}"',
            "X-Saved-Path": str(save_path),
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7070, reload=False)
