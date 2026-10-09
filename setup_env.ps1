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

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Could not activate venv with script" -ForegroundColor Yellow
}

# Upgrade pip
Write-Host "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip" -ForegroundColor Yellow
}
Write-Host "Pip upgraded" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing requirements from requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
    exit 1
}
Write-Host "Requirements installed" -ForegroundColor Green
Write-Host ""

# Install in development mode
Write-Host "Installing ai_service in development mode..."
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install ai_service in dev mode" -ForegroundColor Red
    exit 1
}
Write-Host "ai_service installed in dev mode" -ForegroundColor Green
Write-Host ""

# Test PyTorch installation
Write-Host "Testing PyTorch installation..."
python -c "import torch; print('PyTorch installed')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PyTorch not installed or failed to import" -ForegroundColor Yellow
} else {
    python -c "import torch; print('CUDA available: ' + str(torch.cuda.is_available()))"
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
