#!/bin/bash

# Setup script for TemporalStyleNet
# This script sets up the project environment and downloads necessary resources

set -e  # Exit on error

echo "=================================================="
echo "  TemporalStyleNet Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_success "CUDA detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    print_info "CUDA not detected. Will use CPU (slower)"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies..."
echo "(This may take a few minutes...)"
pip install -r requirements.txt > /dev/null 2>&1
print_success "Dependencies installed"

# Install package in development mode
echo ""
echo "Installing package in development mode..."
pip install -e . > /dev/null 2>&1
print_success "Package installed"

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/videos
mkdir -p data/styles
mkdir -p data/outputs
mkdir -p data/train/content
mkdir -p data/train/styles
mkdir -p checkpoints
mkdir -p logs/tensorboard
mkdir -p tests

# Create .gitkeep files
touch data/videos/.gitkeep
touch data/styles/.gitkeep
touch data/outputs/.gitkeep
touch data/train/content/.gitkeep
touch data/train/styles/.gitkeep

print_success "Directory structure created"

# Download sample style images
echo ""
read -p "Download sample style images? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading sample styles..."
    
    # Create a Python script to download images
    python3 << 'EOF'
import urllib.request
import os

styles = {
    "starry_night.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "great_wave.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/The_Great_Wave_off_Kanagawa.jpg/1280px-The_Great_Wave_off_Kanagawa.jpg",
    "picasso.jpg": "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/PicassoGuernica.jpg/1280px-PicassoGuernica.jpg"
}

for name, url in styles.items():
    try:
        output_path = f"data/styles/{name}"
        if not os.path.exists(output_path):
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded: {name}")
    except Exception as e:
        print(f"Error downloading {name}: {e}")
EOF
    
    print_success "Sample styles downloaded to data/styles/"
fi

# Check FFmpeg
echo ""
echo "Checking FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    print_success "FFmpeg detected"
else
    print_error "FFmpeg not found"
    echo "Please install FFmpeg:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
fi

# Run tests
echo ""
read -p "Run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
        python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
    fi
    print_success "Tests completed"
fi

# Summary
echo ""
echo "=================================================="
echo "  Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Add your video: cp your_video.mp4 data/videos/"
echo "3. Run inference:"
echo "   python scripts/inference.py \\"
echo "     --input data/videos/your_video.mp4 \\"
echo "     --style data/styles/starry_night.jpg \\"
echo "     --output data/outputs/result.mp4"
echo ""
echo "Or launch the demo:"
echo "   python demo/app.py"
echo ""
echo "For more info, see QUICKSTART.md"
echo ""

# Deactivate (user will need to reactivate)
deactivate 2>/dev/null || true
