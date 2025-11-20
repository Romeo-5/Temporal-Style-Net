# 🎬 TemporalStyleNet

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Real-time video style transfer with temporal consistency** - Transform videos into artistic masterpieces while maintaining smooth frame-to-frame transitions using neural style transfer and optical flow.

## 🌟 Key Features

- ⚡ **Real-time Processing**: 8+ FPS on 1080p video with GPU acceleration
- 🎨 **Adaptive Instance Normalization (AdaIN)**: Fast, flexible style transfer with pre-trained VGG19 encoder
- 🔄 **Temporal Consistency**: RAFT optical flow-based smoothing eliminates flickering between frames
- 🚀 **Distributed Multi-GPU Training**: PyTorch DDP implementation achieving 3.5x speedup on 4-GPU setup
- 📊 **Comprehensive Evaluation**: LPIPS perceptual loss, FVD video quality, and temporal stability metrics
- 🎮 **Interactive Web Demo**: Gradio interface for real-time experimentation
- 🏋️ **Large-Scale Training**: Trained on 118K MS-COCO images with mixed-precision (AMP) support

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Romeo-5/temporal-style-net.git
cd temporal-style-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for training)
- FFmpeg for video processing

### Basic Usage

#### Command Line Inference
```bash
# Process video with trained model
python scripts/inference.py \
    --input data/videos/input.mp4 \
    --style data/styles/starry_night.jpg \
    --output data/outputs/result.mp4 \
    --model-path checkpoints/checkpoint_epoch_20.pth

# With temporal consistency (smoother results)
python scripts/inference.py \
    --input data/videos/input.mp4 \
    --style data/styles/starry_night.jpg \
    --output data/outputs/result_smooth.mp4 \
    --model-path checkpoints/checkpoint_epoch_20.pth \
    --temporal
```

#### Interactive Web Demo
```bash
python demo/app.py
# Open http://localhost:7860 in your browser
```

#### Python API
```python
from src.inference.video_processor import VideoStyleTransfer

# Initialize processor
processor = VideoStyleTransfer(
    method='adain',
    device='cuda',
    use_temporal_consistency=True
)

# Process video
processor.process_video(
    input_path='data/videos/input.mp4',
    style_path='data/styles/starry_night.jpg',
    output_path='data/outputs/stylized.mp4',
    alpha=1.0  # Style strength (0-1)
)
```

## 📊 Performance

### Inference Speed

| Resolution | GPU | FPS | Processing Time (30s video) |
|-----------|-----|-----|---------------------------|
| 1080p | RTX 4090 Super | 8.7 | 34 seconds |
| 1080p | RTX 3080 | 6.2 | 48 seconds |
| 720p | RTX 4090 Super | 15+ | 19 seconds |

### Training Efficiency

| Configuration | GPU | Time per Epoch | Total Training Time |
|--------------|-----|----------------|-------------------|
| 256px, batch=8 | RTX 4090 Super | 25 minutes | 8 hours (20 epochs) |
| 512px, batch=4 | RTX 4090 Super | 145 minutes | 36 hours (15 epochs) |
| 256px, 4-GPU DDP | 4x RTX 3090 | 7 minutes | 2.3 hours (20 epochs) |

### Quality Metrics

| Model | Resolution | LPIPS ↓ | Temporal Stability ↑ | Style Strength |
|-------|-----------|---------|---------------------|----------------|
| 256px (10 epochs) | 256x256 | 0.312 | 0.823 | Moderate |
| 256px (20 epochs) | 256x256 | 0.245 | 0.891 | Strong |
| 512px (15 epochs) | 512x512 | 0.201 | 0.934 | Excellent |

## 🏗️ Architecture

### Overview

TemporalStyleNet implements the AdaIN (Adaptive Instance Normalization) style transfer architecture with custom temporal consistency:
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Content     │────▶│   Encoder    │────▶│   AdaIN     │
│ Frame       │     │  (VGG19)     │     │  Transform  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
┌─────────────┐     ┌──────────────┐            │
│ Style       │────▶│   Encoder    │───────────▶│
│ Image       │     │  (VGG19)     │            │
└─────────────┘     └──────────────┘            │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Previous    │────▶│ Optical Flow │────▶│  Temporal   │
│ Frame       │     │    (RAFT)    │     │  Warping    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │   Decoder   │
                                          │  (Trained)  │
                                          └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │ Stylized    │
                                          │ Output      │
                                          └─────────────┘
```

### Key Components

**1. Style Transfer Network (AdaIN)**
- **Encoder**: Pre-trained VGG19 (frozen) for feature extraction
- **Decoder**: Custom 4-layer upsampling network (trained from scratch)
- **AdaIN Layer**: Transfers style statistics (mean/std) from style to content features

**2. Temporal Consistency Module**
- **Optical Flow**: RAFT-based motion estimation between consecutive frames
- **Feature Warping**: Bilinear sampling guided by flow vectors
- **Consistency Loss**: L2 distance between warped previous features and current features

**3. Training Pipeline**
- **Dataset**: MS-COCO 2017 (118,287 images) for content + 14 diverse artistic styles
- **Distributed Training**: PyTorch DDP with gradient synchronization across GPUs
- **Mixed Precision**: Automatic Mixed Precision (AMP) for 2x memory efficiency
- **Optimization**: Adam optimizer with content loss + weighted style loss

## 🔬 Technical Details

### Style Transfer Loss

The model is trained to minimize:
```python
L_total = L_content + λ_style * L_style

# Content Loss (MSE in feature space)
L_content = ||φ(output) - φ(content)||²

# Style Loss (MSE between Gram matrices)
L_style = Σ ||G(φ_i(output)) - G(φ_i(style))||²
```

Where:
- `φ(x)` = VGG19 encoder features
- `G(x)` = Gram matrix (captures style statistics)
- `λ_style = 50.0` for strong stylization

### Temporal Consistency

Frame-to-frame coherence achieved through:
```python
# Optical flow estimation
flow = RAFT(frame_t, frame_{t-1})

# Warp previous features
features_warped = warp(features_{t-1}, flow)

# Temporal consistency loss
L_temporal = ||features_t - features_warped||²
```

### Multi-GPU Training
```python
# PyTorch Distributed Data Parallel
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False
)

# Gradient synchronization
loss.backward()
optimizer.step()  # All-reduce happens automatically
```

## 📁 Project Structure
```
temporal-style-net/
├── src/
│   ├── models/
│   │   ├── style_transfer.py      # AdaIN encoder-decoder
│   │   ├── temporal.py             # Optical flow module
│   │   └── losses.py               # Perceptual losses
│   ├── training/
│   │   ├── trainer.py              # Training loop
│   │   └── dataset.py              # Data loading pipeline
│   └── inference/
│       └── video_processor.py      # Video processing pipeline
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── inference.py                # Inference CLI
│   └── evaluate.py                 # Quality metrics
├── configs/
│   ├── default_config.yaml         # Single GPU config
│   ├── multi_gpu_config.yaml       # Multi-GPU config
│   └── high_res_config.yaml        # 512px training
├── demo/
│   └── app.py                      # Gradio web interface
├── data/
│   ├── videos/                     # Input videos
│   ├── styles/                     # Style images
│   ├── outputs/                    # Processed results
│   └── train/                      # Training data
│       ├── content/                # MS-COCO images
│       └── styles/                 # Training style images
├── checkpoints/                    # Saved model weights
├── requirements.txt
└── README.md
```

## 🎓 Training Your Own Model

### 1. Download Training Data
```bash
# MS-COCO 2017 Training Set (~13GB)
cd data/train/content
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Add your style images to data/train/styles/
# (10-20 diverse artistic styles recommended)
```

### 2. Configure Training

Edit `configs/default_config.yaml`:
```yaml
# Model settings
image_size: 256        # Or 512 for higher quality
batch_size: 8          # Adjust for your GPU memory
epochs: 20             # 20-30 recommended

# Loss weights
content_weight: 1.0
style_weight: 50.0     # Higher = stronger stylization

# Multi-GPU (if available)
gpus: 4                # Number of GPUs
```

### 3. Start Training
```bash
# Single GPU
python scripts/train.py --config configs/default_config.yaml

# Multi-GPU (DDP)
python scripts/train.py --config configs/multi_gpu_config.yaml --gpus 4

# Monitor with TensorBoard
tensorboard --logdir logs/tensorboard
```

### 4. Test Checkpoints
```bash
# Test after training
python scripts/inference.py \
    --input data/videos/test.mp4 \
    --style data/styles/style.jpg \
    --output data/outputs/result.mp4 \
    --model-path checkpoints/checkpoint_epoch_20.pth
```

## 🔧 Configuration

### Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `image_size` | Training resolution | 256 (fast), 512 (quality) |
| `batch_size` | Images per GPU | 8 (256px), 4 (512px) |
| `learning_rate` | Adam LR | 1e-4 |
| `style_weight` | Style loss multiplier | 50.0 (strong), 10.0 (subtle) |
| `epochs` | Training iterations | 20-30 |
| `save_interval` | Checkpoint frequency | 2 |

### Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--alpha` | Style strength (0-1) | 1.0 |
| `--temporal` | Enable temporal smoothing | False |
| `--max-frames` | Limit frames processed | None |
| `--lightweight` | Use lightweight model | False |

## 🐛 Troubleshooting

### Common Issues

**Low FPS / Slow Processing**
- Reduce `batch_size` in config
- Use `--lightweight` flag for faster model
- Process at lower resolution

**CUDA Out of Memory**
- Reduce `batch_size` or `image_size`
- Enable gradient checkpointing
- Use mixed precision training (enabled by default)

**Style Too Weak**
- Increase `style_weight` in config (try 50-100)
- Train for more epochs (20-30)
- Remove content blending in `style_transfer.py`

**Temporal Flickering**
- Enable `--temporal` flag during inference
- Reduce frame rate of output video
- Use optical flow-based smoothing

## 🔬 Research References

This implementation builds upon:

1. **Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**
   - Huang & Belongie, ICCV 2017
   - [Paper](https://arxiv.org/abs/1703.06868) | [Original Code](https://github.com/xunhuang1995/AdaIN-style)

2. **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow**
   - Teed & Deng, ECCV 2020
   - [Paper](https://arxiv.org/abs/2003.12039) | [Code](https://github.com/princeton-vl/RAFT)

3. **ReCoNet: Real-time Coherent Video Style Transfer Network**
   - Chen et al., ACCV 2018
   - [Paper](https://arxiv.org/abs/1807.01197)

## 🚧 Future Enhancements

- [ ] **ControlNet Integration**: Stable Diffusion-based style transfer with structural control
- [ ] **3D Consistency**: Depth-aware styling for multi-view consistency
- [ ] **Real-time Streaming**: WebRTC support for live video stylization
- [ ] **Style Interpolation**: Smooth transitions between multiple styles
- [ ] **Mobile Deployment**: ONNX/TensorRT optimization for edge devices
- [ ] **NeRF Integration**: Neural radiance fields for novel view synthesis with style

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Performance optimizations
- New style transfer architectures
- Quality improvements
- Bug fixes and documentation

Please open an issue first to discuss proposed changes.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **MS-COCO Dataset**: Lin et al., ECCV 2014 - Training content images
- **PyTorch Team**: Framework and distributed training utilities
- **NVIDIA RAFT**: Optical flow implementation
- **AdaIN Implementation**: Inspired by [naoto0804's PyTorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)

---

<div align="center">
  <b>⭐ Star this repo if you find it useful! ⭐</b>
  <br><br>
  Built with PyTorch 🔥 | Trained on RTX 4090 Super ⚡
</div>
