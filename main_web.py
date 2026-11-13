"""Z-Image-Turbo web app – text-to-image generation with FastAPI + static HTML UI."""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from mflux.models.z_image import ZImageTurbo
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model lifecycle – load once at startup, keep in memory
# ---------------------------------------------------------------------------

_model: ZImageTurbo | None = None
_model_loading = False


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
    """Generate an image from the given prompt and parameters."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait a moment and try again.")

    seed = req.seed if req.seed is not None else int(time.time()) % (2**32)

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
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    elapsed = time.time() - start

    # Save to output directory
    if req.filename and req.filename.strip():
        base = req.filename.strip()
        # Ensure .png extension
        if not base.endswith('.png'):
            base += '.png'
        filename = base
    else:
        filename = f"generated_{seed}.png"
    output_path = OUTPUT_DIR / filename
    image.save(output_path)

    # Also return as base64 for inline display
    buf = io.BytesIO()
    image.image.save(buf, format="PNG")
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "url": f"/output/{filename}",
        "base64": b64,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "width": req.width,
        "height": req.height,
    }


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
    uvicorn.run(app, host="0.0.0.0", port=7002)


if __name__ == "__main__":
    main()
