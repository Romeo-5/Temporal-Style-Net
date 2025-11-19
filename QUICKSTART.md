# 🚀 Quick Start Guide

Get up and running with TemporalStyleNet in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- FFmpeg installed

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Romeo-5/temporal-style-net.git
cd temporal-style-net
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Test

### 1. Download sample data

Create the required directories:

```bash
mkdir -p data/videos data/styles data/outputs
```

### 2. Add your files

- Place your video in `data/videos/input.mp4`
- Place style images in `data/styles/` (e.g., `starry_night.jpg`)

### 3. Run inference

```bash
python scripts/inference.py \
    --input data/videos/input.mp4 \
    --style data/styles/starry_night.jpg \
    --output data/outputs/stylized.mp4
```

## Usage Examples

### Basic Style Transfer

```bash
python scripts/inference.py \
    --input video.mp4 \
    --style style.jpg \
    --output result.mp4
```

### Custom Settings

```bash
python scripts/inference.py \
    --input video.mp4 \
    --style monet.jpg \
    --output result.mp4 \
    --alpha 0.8 \
    --lightweight \
    --no-temporal
```

### Test on First 100 Frames

```bash
python scripts/inference.py \
    --input video.mp4 \
    --style style.jpg \
    --output test.mp4 \
    --max-frames 100
```

## Interactive Demo

Launch the web interface:

```bash
python demo/app.py
```

Then open your browser to `http://localhost:7860`

## Training Your Own Model

### 1. Prepare dataset

Download MS COCO (content) and WikiArt (styles):

```bash
# MS COCO train2017
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/train/content/

# WikiArt (or use your own style collection)
# Place style images in data/train/styles/
```

### 2. Start training

Single GPU:

```bash
python scripts/train.py --config configs/default_config.yaml
```

Multi-GPU (4 GPUs):

```bash
python scripts/train.py --config configs/multi_gpu_config.yaml --gpus 4
```

### 3. Monitor training

```bash
tensorboard --logdir logs/tensorboard
```

## Performance Benchmarking

Run comprehensive benchmarks:

```bash
python scripts/benchmark.py \
    --input data/videos/input.mp4 \
    --style data/styles/style.jpg \
    --benchmark all
```

## Project Structure

```
temporal-style-net/
├── src/              # Source code
│   ├── models/       # Model architectures
│   ├── inference/    # Video processing
│   └── training/     # Training utilities
├── scripts/          # Executable scripts
│   ├── train.py      # Training script
│   ├── inference.py  # Inference script
│   └── benchmark.py  # Benchmarking
├── configs/          # Configuration files
├── demo/            # Web demo
├── notebooks/       # Jupyter notebooks
└── data/            # Data directory
```

## Common Issues

### CUDA Out of Memory

Reduce batch size or use lightweight model:

```bash
python scripts/inference.py --input video.mp4 --style style.jpg --output result.mp4 --lightweight
```

### Slow Processing

- Use `--lightweight` flag
- Reduce `--max-frames` for testing
- Enable GPU acceleration

### FFmpeg not found

Install FFmpeg:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Next Steps

- 📖 Read the full [README](README.md)
- 🔬 Explore [notebooks](notebooks/)
- 💻 Check out the [demo](demo/app.py)
- 🚀 Train your own model
- 📊 Run benchmarks

## Support

- Issues: [GitHub Issues](https://github.com/Romeo-5/temporal-style-net/issues)
- Discussions: [GitHub Discussions](https://github.com/Romeo-5/temporal-style-net/discussions)

## Resources

### Style Images

Free style image sources:
- [WikiArt](https://www.wikiart.org/)
- [Google Arts & Culture](https://artsandculture.google.com/)
- [The Met Collection](https://www.metmuseum.org/art/collection)

### Test Videos

Free stock video sources:
- [Pexels](https://www.pexels.com/videos/)
- [Pixabay](https://pixabay.com/videos/)
- [Videvo](https://www.videvo.net/)

---

Happy styling! 🎨✨
