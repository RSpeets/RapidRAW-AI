# RapidRAW AI Service - DirectML Environment Setup Script
# This script sets up a Python virtual environment with DirectML GPU support
# DirectML works with NVIDIA, AMD, and Intel GPUs on Windows

Write-Host "======================================"
Write-Host "RapidRAW AI Service DirectML Setup"
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

# Install DirectML PyTorch
Write-Host "Installing PyTorch with DirectML support..."
Write-Host "Installing torch with DirectML backend..."
& $venvPip install --upgrade torch torchvision torchaudio
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host "PyTorch installed" -ForegroundColor Green
Write-Host ""

# Install torch-directml
Write-Host "Installing torch-directml..."
& $venvPip install torch-directml
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install torch-directml" -ForegroundColor Red
    exit 1
}
Write-Host "torch-directml installed" -ForegroundColor Green
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

# Test PyTorch and DirectML installation
Write-Host "Testing PyTorch and DirectML installation..."
& $venvPython -c "import torch; print('PyTorch installed: ' + torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PyTorch not installed or failed to import" -ForegroundColor Yellow
} else {
    & $venvPython -c "
import torch
import torch_directml

device = torch_directml.device()
print('DirectML device: ' + str(device))
print('DirectML available: True')

# Test tensor operation on DirectML
x = torch.randn(1, 3, 224, 224).to(device)
print('Successfully created tensor on DirectML device')
"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: DirectML not properly configured" -ForegroundColor Yellow
    } else {
        Write-Host "DirectML is available!" -ForegroundColor Green
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
Write-Host "DirectML Setup Notes:"
Write-Host "- DirectML works with NVIDIA, AMD, and Intel GPUs on Windows"
Write-Host "- No separate GPU driver installation required beyond standard drivers"
Write-Host "- For optimal performance, ensure your GPU drivers are up to date"
Write-Host "- DirectML typically provides better performance than CPU mode"
Write-Host "- If you have CUDA-capable NVIDIA GPU, use setup_env_cuda.ps1 for better performance"
Write-Host "- For CPU-only mode, use setup_env.ps1"
Write-Host ""
