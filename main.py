import asyncio
import io
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Global pipeline and lock
pipeline = None
generation_lock = asyncio.Lock()

# Directory for saved outputs
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)


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

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipeline.to(device)

    # Memory optimization for Apple Silicon / consumer hardware
    if device == "mps":
        pipeline.enable_attention_slicing()

    print("✅ Model loaded and ready for inference!")
    yield
    pipeline = None


app = FastAPI(
    title="Text-to-Image Web App",
    description="Generate high quality images from text prompts using SDXL with real-time SSE streaming.",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the desired image")
    filename: Optional[str] = Field("output.png", description="Filename for the saved PNG")
    num_inference_steps: int = Field(30, ge=1, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0, description="Classifier-free guidance scale")


def _run_inference(prompt: str, steps: int, guidance: float, save_path: Path) -> bytes:
    """Synchronous inference function to run in worker thread for direct responses."""
    result = pipeline(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

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

    result = pipeline(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        callback_on_step_end=step_callback,
    ).images[0]

    result.save(save_path, format="PNG")
    return save_path


async def event_generator(prompt: str, steps: int, guidance: float, filename: str):
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
            )
        )

        while not task.done() or not progress_queue.empty():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                yield f"data: {json.dumps(progress)}\n\n"
            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)

        try:
            await task
            completion = {
                "status": "completed",
                "step": steps,
                "total_steps": steps,
                "percentage": 100.0,
                "filename": safe_filename,
                "image_url": f"/images/{safe_filename}",
            }
            yield f"data: {json.dumps(completion)}\n\n"
        except Exception as e:
            error_data = {
                "status": "error",
                "message": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Web interface with prompt, filename, parameters, and interactive SSE stream preview."""
    device, dtype = get_optimal_device_and_dtype()
    device_label = "Apple Silicon (MPS)" if device == "mps" else ("NVIDIA GPU (CUDA)" if device == "cuda" else "CPU")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text to Image Studio</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --input-bg: #0f172a;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }}
        .container {{
            width: 100%;
            max-width: 1000px;
        }}
        header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        h1 {{
            font-size: 2.25rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #a5b4fc, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            display: inline-block;
            background-color: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.4);
            border-radius: 9999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}
        @media (min-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .form-group {{
            margin-bottom: 1.25rem;
        }}
        label {{
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            font-size: 0.95rem;
            color: var(--text-main);
        }}
        .hint {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        textarea, input[type="text"], input[type="number"] {{
            width: 100%;
            background: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-main);
            padding: 0.75rem;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }}
        textarea:focus, input:focus {{
            border-color: var(--accent);
        }}
        textarea {{
            resize: vertical;
            min-height: 110px;
        }}
        .row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        button {{
            width: 100%;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.85rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }}
        button:hover {{
            background: var(--accent-hover);
        }}
        button:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
        }}
        .preview-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 380px;
            background: var(--input-bg);
            border: 2px dashed var(--border);
            border-radius: 0.75rem;
            position: relative;
            overflow: hidden;
            padding: 1rem;
        }}
        .preview-image {{
            max-width: 100%;
            max-height: 480px;
            border-radius: 0.5rem;
            display: none;
            object-fit: contain;
        }}
        .placeholder {{
            color: var(--text-muted);
            text-align: center;
            padding: 2rem;
        }}
        .spinner {{
            display: none;
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s infinite linear;
            margin-bottom: 1rem;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .progress-bar-container {{
            display: none;
            width: 80%;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 9999px;
            height: 8px;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        .progress-bar {{
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #c084fc);
            transition: width 0.2s ease;
        }}
        .status-text {{
            font-size: 0.9rem;
            color: var(--text-muted);
            text-align: center;
        }}
        .download-btn {{
            margin-top: 1rem;
            display: none;
            text-decoration: none;
            background: #10b981;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 0.5rem;
            font-weight: 500;
            text-align: center;
            transition: background 0.2s;
        }}
        .download-btn:hover {{
            background: #059669;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Text to Image Studio</h1>
            <p style="color: var(--text-muted);">Stable Diffusion XL Web Application with Live SSE Stream</p>
            <div class="badge">Hardware Acceleration: {device_label}</div>
        </header>

        <div class="grid">
            <div class="card">
                <form id="gen-form">
                    <div class="form-group">
                        <label for="prompt">Prompt</label>
                        <textarea id="prompt" name="prompt" required placeholder="Describe what you want to see...">A cinematic shot of an astronaut riding a horse on Mars, highly detailed, 8k</textarea>
                    </div>

                    <div class="form-group">
                        <label for="filename">Filename</label>
                        <input type="text" id="filename" name="filename" value="artwork.png" placeholder="e.g. artwork.png">
                        <div class="hint">The generated image will be saved locally & returned with this name.</div>
                    </div>

                    <div class="row">
                        <div class="form-group">
                            <label for="steps">Inference Steps</label>
                            <input type="number" id="steps" name="num_inference_steps" value="30" min="1" max="100">
                            <div class="hint">Higher = more quality (25-40 typical)</div>
                        </div>

                        <div class="form-group">
                            <label for="guidance">Guidance Scale</label>
                            <input type="number" id="guidance" name="guidance_scale" value="7.5" step="0.5" min="0" max="20">
                            <div class="hint">Higher = sticks closer to prompt (7-8 typical)</div>
                        </div>
                    </div>

                    <button type="submit" id="submit-btn">
                        <span>Generate Image</span>
                    </button>
                </form>
            </div>

            <div class="card" style="display: flex; flex-direction: column;">
                <label>Generated Result</label>
                <div class="preview-container">
                    <div class="spinner" id="spinner"></div>
                    <div class="progress-bar-container" id="progress-container">
                        <div class="progress-bar" id="progress-bar"></div>
                    </div>
                    <div class="status-text" id="status-text">Your generated image will appear here</div>
                    <img id="preview" class="preview-image" alt="Generated result" />
                </div>
                <a id="download-link" class="download-btn" download="generated_image.png">Download Image</a>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('gen-form');
        const submitBtn = document.getElementById('submit-btn');
        const spinner = document.getElementById('spinner');
        const progressContainer = document.getElementById('progress-container');
        const progressBar = document.getElementById('progress-bar');
        const statusText = document.getElementById('status-text');
        const preview = document.getElementById('preview');
        const downloadLink = document.getElementById('download-link');

        form.addEventListener('submit', async (e) => {{
            e.preventDefault();

            const prompt = document.getElementById('prompt').value;
            const filename = document.getElementById('filename').value;
            const steps = parseInt(document.getElementById('steps').value);
            const guidance = parseFloat(document.getElementById('guidance').value);

            // UI loading state
            submitBtn.disabled = true;
            spinner.style.display = 'block';
            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            statusText.style.display = 'block';
            statusText.textContent = 'Connecting to server stream...';
            preview.style.display = 'none';
            downloadLink.style.display = 'none';

            try {{
                const formData = new FormData();
                formData.append('prompt', prompt);
                formData.append('filename', filename);
                formData.append('num_inference_steps', steps);
                formData.append('guidance_scale', guidance);

                const response = await fetch('/generate-stream', {{
                    method: 'POST',
                    body: formData
                }});

                if (!response.ok) {{
                    throw new Error('Server error: ' + response.status + ' ' + response.statusText);
                }}

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, {{ stream: true }});
                    const lines = buffer.split('\\n');
                    buffer = lines.pop(); // keep last incomplete line

                    for (const line of lines) {{
                        const trimmed = line.trim();
                        if (trimmed.startsWith('data: ')) {{
                            const data = JSON.parse(trimmed.substring(6));

                            if (data.status === 'queued') {{
                                statusText.textContent = data.message || 'Queued for generation...';
                            }} else if (data.status === 'starting') {{
                                statusText.textContent = data.message || 'Starting inference...';
                                progressBar.style.width = '2%';
                            }} else if (data.status === 'generating') {{
                                statusText.textContent = `Denoising step ${{data.step}} of ${{data.total_steps}} (${{data.percentage}}%)`;
                                progressBar.style.width = `${{data.percentage}}%`;
                            }} else if (data.status === 'completed') {{
                                progressBar.style.width = '100%';
                                statusText.textContent = 'Generation complete!';
                                
                                // Display image preview
                                preview.src = data.image_url + '?t=' + new Date().getTime();
                                preview.style.display = 'block';
                                spinner.style.display = 'none';
                                progressContainer.style.display = 'none';

                                downloadLink.href = data.image_url;
                                downloadLink.download = data.filename;
                                downloadLink.style.display = 'inline-block';
                            }} else if (data.status === 'error') {{
                                throw new Error(data.message || 'Unknown generation error');
                            }}
                        }}
                    }}
                }}
            }} catch (err) {{
                spinner.style.display = 'none';
                progressContainer.style.display = 'none';
                statusText.style.display = 'block';
                statusText.textContent = 'Error: ' + err.message;
            }} finally {{
                submitBtn.disabled = false;
            }}
        }});
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


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


@app.post("/generate-stream")
async def generate_stream_form(
    prompt: str = Form(...),
    filename: str = Form("output.png"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
):
    """Form endpoint returning an SSE stream of generation status and final image URL."""
    return StreamingResponse(
        event_generator(
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance_scale,
            filename=filename,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
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
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate", response_class=Response)
async def generate_image(
    prompt: str = Form(...),
    filename: str = Form("output.png"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
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
