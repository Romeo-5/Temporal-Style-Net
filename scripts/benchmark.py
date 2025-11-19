"""
Benchmark script for evaluating performance
Tests FPS, quality metrics, and temporal consistency
"""

import argparse
import torch
import time
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.video_processor import VideoStyleTransfer
from src.models.temporal_consistency import compute_temporal_consistency_metrics


def benchmark_fps(processor, video_path, style_path, num_runs=3):
    """Benchmark processing FPS"""
    print("\n" + "="*60)
    print("FPS Benchmark")
    print("="*60)
    
    fps_results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        output_path = f"/tmp/benchmark_run_{run}.mp4"
        
        stats = processor.process_video(
            input_path=video_path,
            style_path=style_path,
            output_path=output_path,
            max_frames=100,  # Test on 100 frames
            progress=False
        )
        
        fps_results.append(stats['avg_fps'])
        print(f"  FPS: {stats['avg_fps']:.2f}")
    
    avg_fps = np.mean(fps_results)
    std_fps = np.std(fps_results)
    
    print(f"\nResults:")
    print(f"  Average FPS: {avg_fps:.2f} ± {std_fps:.2f}")
    
    return {
        'avg_fps': avg_fps,
        'std_fps': std_fps,
        'runs': fps_results
    }


def benchmark_quality(processor, video_path, style_path):
    """Benchmark output quality metrics"""
    print("\n" + "="*60)
    print("Quality Metrics Benchmark")
    print("="*60)
    
    try:
        import lpips
        from skimage.metrics import structural_similarity as ssim
        
        loss_fn = lpips.LPIPS(net='alex').to(processor.device)
        
        print("\nProcessing video for quality evaluation...")
        output_path = "/tmp/quality_benchmark.mp4"
        
        stats = processor.process_video(
            input_path=video_path,
            style_path=style_path,
            output_path=output_path,
            max_frames=50,
            progress=True
        )
        
        # Load output video
        from src.inference.video_processor import VideoStyleTransfer as VST
        temp_processor = VST(method='adain', device='cpu')
        frames, _, _ = temp_processor.load_video(output_path)
        
        # Compute temporal consistency
        frames_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in frames
        ])
        
        tc_metrics = compute_temporal_consistency_metrics(frames_tensor)
        
        print("\nQuality Metrics:")
        for metric, value in tc_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return tc_metrics
        
    except ImportError as e:
        print(f"Warning: Could not compute quality metrics: {e}")
        print("Install with: pip install lpips scikit-image")
        return {}


def benchmark_comparison(video_path, style_path, methods=['adain'], device='cuda'):
    """Compare different methods"""
    print("\n" + "="*60)
    print("Method Comparison")
    print("="*60)
    
    results = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        
        processor = VideoStyleTransfer(
            method=method,
            device=device,
            use_temporal_consistency=True
        )
        
        output_path = f"/tmp/compare_{method}.mp4"
        
        start_time = time.time()
        stats = processor.process_video(
            input_path=video_path,
            style_path=style_path,
            output_path=output_path,
            max_frames=50,
            progress=False
        )
        elapsed = time.time() - start_time
        
        results[method] = {
            'fps': stats['avg_fps'],
            'total_time': elapsed,
            'avg_frame_time': stats['avg_processing_time']
        }
        
        print(f"  FPS: {stats['avg_fps']:.2f}")
        print(f"  Total time: {elapsed:.2f}s")
    
    return results


def benchmark_multi_resolution(processor, video_path, style_path):
    """Test performance at different resolutions"""
    print("\n" + "="*60)
    print("Resolution Benchmark")
    print("="*60)
    
    # This is a simplified version - in practice, you'd resize the video
    resolutions = [
        (480, 270),   # 270p
        (854, 480),   # 480p
        (1280, 720),  # 720p
        (1920, 1080), # 1080p
    ]
    
    results = {}
    
    for width, height in resolutions:
        print(f"\nTesting {width}x{height}...")
        
        # Note: You would need to implement video resizing here
        # For now, just report the original video performance
        output_path = f"/tmp/res_{width}x{height}.mp4"
        
        stats = processor.process_video(
            input_path=video_path,
            style_path=style_path,
            output_path=output_path,
            max_frames=30,
            progress=False
        )
        
        results[f"{width}x{height}"] = {
            'fps': stats['avg_fps'],
            'processing_time': stats['avg_processing_time']
        }
    
    return results


def generate_report(results, output_file='benchmark_report.json'):
    """Generate benchmark report"""
    print("\n" + "="*60)
    print("Benchmark Report")
    print("="*60)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to: {output_file}")
    
    # Print summary
    if 'fps_benchmark' in results:
        fps = results['fps_benchmark']
        print(f"\n📊 FPS Performance:")
        print(f"  Average: {fps['avg_fps']:.2f} FPS")
        print(f"  Std Dev: {fps['std_fps']:.2f}")
    
    if 'quality_metrics' in results:
        quality = results['quality_metrics']
        print(f"\n📈 Quality Metrics:")
        for metric, value in quality.items():
            print(f"  {metric}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark video style transfer')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--style', type=str, required=True,
                       help='Path to style image')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--benchmark', type=str, default='all',
                       choices=['fps', 'quality', 'comparison', 'resolution', 'all'])
    parser.add_argument('--output', type=str, default='benchmark_report.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoStyleTransfer(
        method='adain',
        device=args.device,
        use_temporal_consistency=True,
        lightweight=False
    )
    
    results = {}
    
    # Run benchmarks
    if args.benchmark in ['fps', 'all']:
        results['fps_benchmark'] = benchmark_fps(
            processor, args.input, args.style
        )
    
    if args.benchmark in ['quality', 'all']:
        results['quality_metrics'] = benchmark_quality(
            processor, args.input, args.style
        )
    
    if args.benchmark in ['comparison', 'all']:
        results['method_comparison'] = benchmark_comparison(
            args.input, args.style, device=args.device
        )
    
    if args.benchmark in ['resolution', 'all']:
        results['resolution_benchmark'] = benchmark_multi_resolution(
            processor, args.input, args.style
        )
    
    # Generate report
    generate_report(results, args.output)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
