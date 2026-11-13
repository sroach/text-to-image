"""Z-Image-Turbo web app – text-to-image generation with FastAPI + static HTML UI."""

import asyncio
import base64
import io
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from mflux.models.z_image import ZImageTurbo
from mflux.models.common.config.config import Config
from pydantic import BaseModel, Field
import mlx.core as mx


# ---------------------------------------------------------------------------
# Model lifecycle – load once at startup, keep in memory
# ---------------------------------------------------------------------------

_model: ZImageTurbo | None = None
_model_loading = False


# ---------------------------------------------------------------------------
# SSE progress infrastructure
# ---------------------------------------------------------------------------

# Per-request progress queues: request_id -> asyncio.Queue
_progress_queues: dict[str, asyncio.Queue] = {}

# Completed results: request_id -> result dict or error info
_results: dict[str, dict[str, Any]] = {}


class ProgressCallback:
    """Reports each inference step to the SSE queue for the current request."""

    def __init__(self, request_id: str, total_steps: int):
        self.request_id = request_id
        self.total_steps = total_steps

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: Any,
    ) -> None:
        q = _progress_queues.get(self.request_id)
        if q is not None:
            progress = int((t + 1) / self.total_steps * 100)
            try:
                q.put_nowait(f"data: {progress}\n\n")
            except asyncio.QueueFull:
                pass  # client disconnected or slow


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_loading
    print("Loading Z-Image-Turbo model (first run downloads weights from Hugging Face)...")
    _model_loading = True
    # quantize=4 → ~8 GB RAM, good quality on 32 GB Mac; use quantize=8 for 48 GB+ Macs
    _model = ZImageTurbo(quantize=4)
    _model_loading = False
    print("Model loaded and ready.")
    yield


app = FastAPI(title="Z-Image-Turbo Web", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="Text prompt describing the image")
    negative_prompt: str = Field("", description="Negative prompt (note: Z-Image-Turbo ignores CFG, but field is kept for convenience)")
    width: int = Field(1024, ge=256, le=2048, multiple_of=16, description="Image width (multiple of 16)")
    height: int = Field(1024, ge=256, le=2048, multiple_of=16, description="Image height (multiple of 16)")
    num_inference_steps: int = Field(9, ge=1, le=30, description="Inference steps (8-12 recommended)")
    seed: int | None = Field(None, ge=0, description="Random seed (leave empty for random)")
    filename: str | None = Field(None, description="Custom filename (without extension; leave empty for auto)")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).parent / "templates"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    html = (TEMPLATES_DIR / "index.html").read_text()
    return HTMLResponse(html)


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Start image generation and return a request_id for SSE progress tracking."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait a moment and try again.")

    request_id = str(uuid.uuid4())
    seed = req.seed if req.seed is not None else int(time.time()) % (2**32)

    # Set up SSE queue for this request
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _progress_queues[request_id] = q
    q.put_nowait("data: 0\n\n")  # initial progress

    # Register progress callback
    callback = ProgressCallback(request_id, req.num_inference_steps)
    _model.callbacks.register(callback)

    async def run_generation():
        start = time.time()
        try:
            image = _model.generate_image(
                seed=seed,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
            )
        except Exception as exc:
            # Store error result
            _results[request_id] = {"error": str(exc)}
            try:
                q.put_nowait(f'data: {{"error": "{exc}"}}\n\n')
            except asyncio.QueueFull:
                pass
            _model.callbacks.in_loop.remove(callback)
            _progress_queues.pop(request_id, None)
            return

        elapsed = time.time() - start

        # Save to output directory
        if req.filename and req.filename.strip():
            base = req.filename.strip()
            if not base.endswith('.png'):
                base += '.png'
            filename = base
        else:
            filename = f"generated_{seed}.png"
        output_path = OUTPUT_DIR / filename
        image.save(output_path)

        # Encode as base64 for inline display
        buf = io.BytesIO()
        image.image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        result = {
            "url": f"output/{filename}",
            "base64": b64,
            "seed": seed,
            "elapsed_seconds": round(elapsed, 2),
            "width": req.width,
            "height": req.height,
        }
        _results[request_id] = result

        # Signal completion
        try:
            q.put_nowait("data: 100\n\n")
            q.put_nowait("data: [DONE]\n\n")
        except asyncio.QueueFull:
            pass

        # Cleanup
        _model.callbacks.in_loop.remove(callback)
        _progress_queues.pop(request_id, None)

    asyncio.create_task(run_generation())

    return {"request_id": request_id}


@app.get("/progress/{request_id}")
async def progress(request_id: str):
    """SSE endpoint: streams progress updates for the given request."""
    q = _progress_queues.get(request_id)
    if q is None:
        raise HTTPException(status_code=404, detail="Request not found or already completed")

    async def event_stream():
        try:
            while True:
                message = await asyncio.wait_for(q.get(), timeout=60)
                yield message
                if message == "data: [DONE]\n\n":
                    break
        except asyncio.TimeoutError:
            yield 'data: {"error": "Generation timed out"}\n\n'
        except asyncio.CancelledError:
            pass
        finally:
            _progress_queues.pop(request_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


@app.get("/result/{request_id}")
async def result(request_id: str):
    """Poll endpoint: returns the final result once generation is complete."""
    res = _results.get(request_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Result not ready yet")
    if "error" in res:
        raise HTTPException(status_code=500, detail=res["error"])
    # Clean up completed result
    _results.pop(request_id, None)
    return res


@app.get("/output/{filename}")
async def get_output(filename: str):
    """Serve a generated image from the output directory."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


def main():
    """Entry point: start the uvicorn server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7002, root_path="/text-2-image")


if __name__ == "__main__":
    main()
