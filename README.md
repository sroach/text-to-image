# Text-to-Image Web App

Generate high-quality images from text prompts using **Stable Diffusion XL (SDXL)** via a FastAPI web application.

## Features

- **SDXL Model**: Uses `stabilityai/stable-diffusion-xl-base-1.0` for high-quality image generation
- **Multi-Platform Support**: Runs on Apple Silicon (MPS), NVIDIA GPUs (CUDA), or CPU
- **Web UI**: Browser-based interface for prompt input and image viewing
- **REST API**: Programmatic access with JSON request/response
- **Real-time Streaming**: Server-Sent Events (SSE) for progress updates during generation
- **Apple Silicon Optimized**: Uses fixed VAE (`madebyollin/sdxl-vae-fp16-fix`) for proper float16 support on MPS

## Prerequisites

- **Python 3.11+**
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/))
- **Hugging Face account** with access to `stabilityai/stable-diffusion-xl-base-1.0`
- **Sufficient RAM/GPU memory**: ~10GB+ recommended for the SDXL model

## Setup

1. **Set your Hugging Face token** (required for model access):
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```

2. **Sync dependencies**:
   ```bash
   uv sync
   ```

## Usage

### Quick Start

Run the application using the provided script:
```bash
./run.sh
```

Or manually:
```bash
uv run main.py
```

On first run, the SDXL model will be downloaded and loaded into memory (this may take a while).

### Accessing the App

- **Web UI**: Open `http://localhost:8000` in your browser
- **API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

### API Endpoints

#### Generate Image (POST)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 30,
    "guidance_scale": 7.5
  }'
```

**Request Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | (required) | Text description of the desired image |
| `negative_prompt` | string | null | Elements to exclude from the image |
| `num_inference_steps` | int | 30 | Number of denoising steps (1-100) |
| `guidance_scale` | float | 7.5 | Classifier-free guidance scale (0-20) |
| `width` | int | 1024 | Image width (256-2048) |
| `height` | int | 1024 | Image height (256-2048) |
| `filename` | string | "output.png" | Filename for saved image |

## Project Structure

```
text-to-image/
├── main.py            # Main FastAPI application
├── main_web.py        # Web interface version
├── run.sh             # Quick start script
├── pyproject.toml     # Project dependencies
├── uv.lock            # Locked dependencies
├── templates/         # Jinja2 HTML templates
└── generated_images/  # Output directory for generated images
```

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **mflux**: ML framework
- **PyTorch**: Deep learning library (auto-installed via mflux)
- **diffusers**: Hugging Face diffusion models library

## Notes

- The model is loaded once into memory on startup and reused for all requests
- Concurrent generation requests are handled via an async lock to prevent conflicts
- Generated images are saved in the `generated_images/` directory
- On Apple Silicon Macs, the app automatically uses the MPS backend with a patched VAE for proper float16 support

## License

This project uses models from Hugging Face that may have their own licensing terms. Please review the [SDXL license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for usage guidelines.
