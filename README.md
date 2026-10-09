# RapidRAW AI Service

A high-performance Python FastAPI backend for AI-powered image processing in [RapidRAW](https://github.com/RSpeets/RapidRAW). Provides intelligent image inpainting with smart cropping for full-resolution compositing.

## Features

- **Smart Inpainting**: Stable Diffusion-based image inpainting with intelligent cropping for improved quality
- **Full-Resolution Output**: Crops mask region → processes at 512x512 → composites result back at full resolution
- **GPU Acceleration**: CUDA support for 10-50x faster inference (automatic CPU fallback)
- **Production-Ready API**: Async FastAPI with middleware and cloud-style endpoints
- **Health Monitoring**: Built-in health checks and model status endpoints
- **Debug Output**: Automatic pipeline visualization (input, mask, output at each stage)

## Quick Start

### Choose Your Setup

Three setup scripts available for different scenarios:

| Script | Best For | GPU Types | Installation Time |
|--------|----------|-----------|-------------------|
| **setup_env_cpu.ps1** | Development/testing | None (CPU only) | ~2-3 min |
| **setup_env_cuda.ps1** | Production (NVIDIA) | NVIDIA GPUs only | ~5-10 min |
| **setup_env_directml.ps1** | Any GPU on Windows | NVIDIA/AMD/Intel | ~3-5 min |

---

### Setup Option 1: CPU Only (Fastest Setup)

Best for testing and development without GPU:

```powershell
cd E:\Python\RapidRAW-AI
.\setup_env_cpu.ps1
```

- Works on any Windows PC  
- Slower inference (1-2 minutes per image)

---

### Setup Option 2: CUDA (NVIDIA GPUs) — **Recommended for Production**

For 10-50x faster inference with NVIDIA GPUs:

```powershell
cd E:\Python\RapidRAW-AI
.\setup_env_cuda.ps1
```

**Prerequisites:**
- NVIDIA GPU (GeForce RTX, RTX A-series, H-series, L-series, etc.)
- NVIDIA drivers installed (check: run `nvidia-smi` in PowerShell)

**Check your NVIDIA GPU:**
```powershell
nvidia-smi
```

If this command fails, install the latest NVIDIA drivers: https://www.nvidia.com/Download/driverDetails.aspx

**Optional: Install CUDA Toolkit**

Visit https://developer.nvidia.com/cuda-downloads and select:
- OS: Windows
- Architecture: x86_64
- Version: 12.x or 11.8
- Installer type: exe (local)

*(The `setup_env_cuda.ps1` script will automatically install PyTorch with CUDA 12.1 support, with fallback to CUDA 11.8)*

---

### Setup Option 3: DirectML (Any GPU on Windows)

For AMD, Intel, or NVIDIA GPUs on Windows without installing CUDA:

```powershell
cd E:\Python\RapidRAW-AI
.\setup_env_directml.ps1
```

- Works with NVIDIA, AMD, Intel GPUs  
- No CUDA toolkit installation needed  
- Simpler setup than CUDA  
- Slightly slower than CUDA on NVIDIA GPUs (~10-20%)

**Best if:**
- You have an AMD Radeon or Intel Arc/UHD GPU
- You want GPU acceleration without installing CUDA
- You want maximum compatibility across GPU types

---

### Summary Table: Performance & Setup

| Option | GPU Speed | Setup Complexity | NVIDIA Only |
|--------|-----------|------------------|------------|
| CPU | 1x (baseline) | Easiest | No |
| DirectML | ~15-20x | Easy | No |
| CUDA | ~20-25x | Medium | Yes |

### Activate Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Start the Service

```powershell
python -m ai_service
```

You should see:
```
INFO:     Started server process [xxxxx]
INFO:ai_service.main:AI Service starting up...
INFO:ai_service.models_manager:Using device: cuda/cpu/directml
INFO:ai_service.main:AI Service ready
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**First Run Note:**
On first run with GPU, models will be downloaded and cached (~4GB). Subsequent runs load instantly.

### Verify It Works

```powershell
# In another terminal
curl http://127.0.0.1:8000/health
```

Response:
```json
{"status":"healthy","version":"0.1.0"}
```

**Check which device is being used:**
```powershell
curl http://127.0.0.1:8000/models/status
```

Look for your device type in the response (e.g., `"device": "cuda"` or `"device": "directml"`).

**Done!** Service is running.

## API Endpoints

### Health & Status

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check service health |
| `/models/status` | GET | Get available models and device info |

### Image Inpainting

| Endpoint | Method | Style | Use Case |
|----------|--------|-------|----------|
| `/inpaint` | POST | Middleware | RapidRAW integration (uses cached source_id) |
| `/inpaint/cloud` | POST | Cloud | Direct image_base64 in request |
| `/upload_source` | POST | Upload | Pre-upload source image for caching |

### Request Examples

**Middleware-style inpainting (RapidRAW uses this):**
```python
# 1. Upload source image
curl -X POST http://127.0.0.1:8000/upload_source \
  -F "file=@source.jpg" \
  -F "source_id=abc123"

# 2. Request inpainting
curl -X POST http://127.0.0.1:8000/inpaint \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "abc123",
    "mask_base64": "iVBORw0KGgo...",
    "prompt": "yellow flower",
    "negative_prompt": "blur, distortion",
    "seed": 0,
    "inference_steps": 50,
    "guidance_scale": 7.5
  }'
```

**Response:**
```json
{
  "result_base64": "iVBORw0KGgo...",
  "size": [6240, 4160],
  "model_type": "inpaint",
  "inference_time_ms": 32000
}
```

## Configuration

Set environment variables before running:

```powershell
# Use GPU (recommended, default if available)
$env:AI_DEVICE = "cuda"

# Or force CPU
$env:AI_DEVICE = "cpu"

# Service port (default: 8000)
$env:AI_PORT = "8000"

# Number of inference steps (default: 50, lower = faster but lower quality)
$env:AI_INFERENCE_STEPS = "50"
```

Or create `.env` file:
```ini
AI_DEVICE=cuda
AI_PORT=8000
AI_INFERENCE_STEPS=50
```

## Architecture

### Smart Cropping Pipeline

The inpainting model has a hard 512x512 pixel limit. Instead of scaling the entire large image, the service:

1. **Find mask region** - Calculate bounding box of inpainting mask
2. **Expand by 30%** - Add context around mask
3. **Make square** - Use larger dimension, round to multiple of 64
4. **Crop from original** - Extract this region at full resolution
5. **Scale to 512x512** - Feed to inpainting model
6. **Get result** - 512x512 output from model
7. **Scale back** - Restore to original crop size
8. **Composite onto original** - Blend using mask as alpha channel

**Result**: Full-resolution output with AI-improved region seamlessly blended into original.

### Key Files

- **`ai_service/main.py`** - FastAPI endpoints and request routing
- **`ai_service/models_manager.py`** - Model loading, inference, and smart cropping logic (lines 285-435)
- **`ai_service/handlers.py`** - Request processing and image cache management
- **`ai_service/config.py`** - Configuration and settings
- **`ai_service/models.py`** - Pydantic request/response schemas

### Debug Output

Each inpainting request saves to `inpaint_debug/` folder:
- `01_source_512.png` - Cropped and scaled input to AI
- `02_mask_512.png` - Scaled mask used by model
- `03_result_raw_512.png` - Raw 512x512 output from model
- `04_result_scaled.png` - Scaled back to original crop size
- `parameters.txt` - All parameters and sizes used

## Performance

### With GPU (NVIDIA CUDA - RTX 3070 Ti)
- **Inpainting**: 30-60 seconds (50 steps, 512x512)
- **Cold start** (model load): 15-20 seconds
- **Warm (cached model)**: 30-35 seconds

### With CPU
- **Inpainting**: 3-5 minutes (very slow)
- **Not recommended for production**

**Tip**: GPU is 10-50x faster. For best results, use a modern NVIDIA GPU.

## Testing

Run tests:

```powershell
# Activate environment first
.\.venv\Scripts\Activate.ps1

# Run all tests
pytest -v

# Run specific test
pytest tests/test_api.py::test_health -v

# Run with coverage
pytest --cov=ai_service --cov-report=html
```

Current test coverage:
- Health check endpoint
- Model status endpoint
- Image upload and caching
- Inpainting request handling
- Error responses

## Troubleshooting

### CUDA Setup Issues

#### CUDA not available / GPU not detected

**Symptom:** Script shows `CUDA available: False` or you're getting slow CPU-only inference

**Step 1: Verify your GPU and drivers**
```powershell
# Check if your GPU and drivers are detected
nvidia-smi
```

If this fails:
- Install NVIDIA drivers: https://www.nvidia.com/Download/driverDetails.aspx
- Restart your computer after driver installation

**Step 2: Install/Update CUDA Toolkit**
Visit https://developer.nvidia.com/cuda-downloads and select:
- OS: Windows
- Architecture: x86_64  
- Version: 12.x (recommended) or 11.8
- Installer type: exe (local)

**Step 3: Reinstall PyTorch with CUDA support**
```powershell
# Activate your environment
.\.venv\Scripts\Activate.ps1

# Reinstall with CUDA 12.1
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If that fails, try CUDA 11.8
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: Verify CUDA is working**
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

#### CUDA version mismatch

**Symptom:** "CUDA runtime error" or "driver version is insufficient"

**Solution:** Check your NVIDIA driver supports CUDA 12.1:
```powershell
nvidia-smi
# Look at "CUDA Version" row (e.g., "12.1", "11.8")
```

Match your driver's CUDA version in the setup script or reinstall PyTorch with the correct version:
```powershell
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Out of memory error during inference

**Symptom:** "CUDA out of memory" error during inpainting

**Solutions:**
```powershell
# 1. Use fewer inference steps (trades speed for quality)
$env:AI_INFERENCE_STEPS = "30"
python -m ai_service

# 2. Reduce inference resolution (if supported)
$env:AI_INFERENCE_RESOLUTION = "384"
python -m ai_service

# 3. Use CPU instead (slower but uses system RAM)
$env:AI_DEVICE = "cpu"
python -m ai_service

# 4. Close other GPU-using applications (Chrome, games, other AI tools)
```

### General Troubleshooting

#### "Module not found" error

```powershell
# Ensure you're in the virtual environment
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

#### Service won't start

```powershell
# Check Python version (need 3.9+)
python --version

# Verify PyTorch installation
python -c "import torch; print('PyTorch:', torch.__version__)"

# View full error log
python -m ai_service
```

#### Slow inference (CPU fallback)

If inference is very slow (5+ minutes):

```powershell
# Verify you're actually using GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, see "CUDA not available" section above
# If True, check GPU utilization while running
# Open Task Manager → Performance → GPU (should be >50%)
```

#### Port already in use

If you get "Address already in use":
```powershell
# Change the port
$env:AI_PORT = "8001"
python -m ai_service

# Or kill the existing process
Get-Process python | Where-Object {$_.CommandLine -like "*ai_service*"} | Stop-Process
```

## Integration with RapidRAW

Once the service is running at `http://127.0.0.1:8000`, RapidRAW automatically uses:

- `/inpaint` - For inpainting operations when you use the brush tool
- `/upload_source` - For efficient image caching

**In RapidRAW settings:**
```
AI Service URL: http://127.0.0.1:8000
```

## Project Structure

```
RapidRAW-AI/
├── ai_service/              # Main service package
│   ├── __main__.py          # Entry point (python -m ai_service)
│   ├── main.py              # FastAPI app and endpoints
│   ├── models_manager.py    # Model inference logic
│   ├── handlers.py          # Request processing
│   ├── config.py            # Settings
│   └── models.py            # Pydantic schemas
├── tests/                   # Unit and integration tests
│   └── test_api.py          # API endpoint tests
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package metadata
├── pytest.ini               # Test configuration
├── setup_env_*.ps1          # Windows setup scripts
└── README.md                # This file
```

## Python & Dependencies

**Requirements:**
- Python 3.9+
- NVIDIA GPU with CUDA support (optional but recommended)
- 8GB+ VRAM (for model + inference)

**Core Dependencies:**
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pillow** - Image processing
- **PyTorch** - Deep learning framework
- **Diffusers** - Stable Diffusion pipeline
- **Transformers** - Model loading from Hugging Face
- **Pydantic** - Request/response validation

See `requirements.txt` for complete list with versions.

## Deployment

### Local Development

```powershell
.\.venv\Scripts\Activate.ps1
python -m ai_service
```

### Production Deployment

For production environments:

```powershell
# Use multiple workers with gunicorn
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker `
  --bind 0.0.0.0:8000 `
  --timeout 300 `
  ai_service:app
```

Or use Docker (if applicable to your setup).

## License

MIT - See LICENSE file for details

## Contributing

Contributions welcome! This is part of the RapidRAW project.

For issues or feature requests, please open an issue on GitHub.

## Questions?

- Check [Troubleshooting](#-troubleshooting) above
- Review documentation in `/md` folder
- Check service logs when running: `python -m ai_service`
- Inspect debug output in `inpaint_debug/` folder
