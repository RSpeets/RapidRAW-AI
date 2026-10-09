# RapidRAW AI Service - CUDA Environment Setup Script
# This script sets up a Python virtual environment with CUDA-enabled GPU support

Write-Host "======================================"
Write-Host "RapidRAW AI Service CUDA Setup"
Write-Host "======================================"
Write-Host ""

# Check if Python is installed
$pythonCheck = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "Found Python: $pythonCheck" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "Virtual environment created" -ForegroundColor Green
Write-Host ""

# Use venv Python for all subsequent commands
$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

# Upgrade pip
Write-Host "Upgrading pip, setuptools, and wheel..."
& $venvPip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip" -ForegroundColor Yellow
}
Write-Host "Pip upgraded" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing requirements from requirements.txt..."
& $venvPip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
    exit 1
}
Write-Host "Requirements installed" -ForegroundColor Green
Write-Host ""

# Install CUDA-enabled PyTorch
Write-Host "Installing CUDA-enabled PyTorch..."
Write-Host "Installing torch, torchvision with CUDA 12.1 support..."
& $venvPip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to install CUDA-enabled PyTorch" -ForegroundColor Yellow
    Write-Host "Attempting fallback CUDA 11.8 version..."
    & $venvPip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install PyTorch with both CUDA versions" -ForegroundColor Red
        exit 1
    }
}
Write-Host "CUDA-enabled PyTorch installed" -ForegroundColor Green
Write-Host ""

# Install compatible diffusers and transformers versions
Write-Host "Installing compatible diffusers and transformers versions..."
& $venvPip install "diffusers>=0.24.0,<0.30.0" "transformers>=4.35.0,<4.45.0"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install compatible diffusers/transformers" -ForegroundColor Red
    exit 1
}
Write-Host "Compatible versions installed" -ForegroundColor Green
Write-Host ""

# Install in development mode
Write-Host "Installing ai_service in development mode..."
& $venvPip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install ai_service in dev mode" -ForegroundColor Red
    exit 1
}
Write-Host "ai_service installed in dev mode" -ForegroundColor Green
Write-Host ""

# Test PyTorch and CUDA installation
Write-Host "Testing PyTorch and CUDA installation..."
& $venvPython -c "import torch; print('PyTorch installed: ' + torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PyTorch not installed or failed to import" -ForegroundColor Yellow
} else {
    & $venvPython -c "import torch; print('CUDA available: ' + str(torch.cuda.is_available())); print('CUDA device count: ' + str(torch.cuda.device_count())); cuda_available = torch.cuda.is_available(); exit(0 if cuda_available else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: CUDA not detected. Ensure NVIDIA drivers are installed and CUDA toolkit is properly configured." -ForegroundColor Yellow
        Write-Host "Visit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
    } else {
        Write-Host "CUDA is available!" -ForegroundColor Green
        & $venvPython -c "import torch; print('GPU: ' + torch.cuda.get_device_name(0))"
    }
}

Write-Host ""
Write-Host "======================================"
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "======================================"
Write-Host ""
Write-Host "To activate the environment in the future:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To run tests:"
Write-Host "pytest -v"
Write-Host ""
Write-Host "To start the service:"
Write-Host "python -m ai_service"
Write-Host ""
Write-Host "CUDA Setup Notes:"
Write-Host "- PyTorch was installed with CUDA 12.1 support (with fallback to CUDA 11.8)"
Write-Host "- Ensure your NVIDIA GPU drivers are up to date"
Write-Host "- Verify CUDA toolkit installation: https://developer.nvidia.com/cuda-downloads"
Write-Host ""
