# RapidRAW AI Service - Environment Setup Script
# This script sets up a separate Python virtual environment with all dependencies

Write-Host "======================================"
Write-Host "RapidRAW AI Service Environment Setup"
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

# Install CPU PyTorch
Write-Host "Installing PyTorch (CPU)..."
& $venvPip install torch torchvision torchaudio
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host "PyTorch installed" -ForegroundColor Green
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

# Test PyTorch installation
Write-Host "Testing PyTorch installation..."
& $venvPython -c "import torch; print('PyTorch installed')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PyTorch not installed or failed to import" -ForegroundColor Yellow
} else {
    & $venvPython -c "import torch; print('CUDA available: ' + str(torch.cuda.is_available()))"
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
