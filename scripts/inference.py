"""
Inference script for video style transfer
Easy command-line interface for processing videos
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.video_processor import VideoStyleTransfer, create_side_by_side_comparison


def main():
    parser = argparse.ArgumentParser(
        description='Apply style transfer to videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/inference.py --input video.mp4 --style starry_night.jpg --output result.mp4
  
  # With custom settings
  python scripts/inference.py --input video.mp4 --style monet.jpg --output result.mp4 \\
      --alpha 0.8 --no-temporal --lightweight
  
  # Process first 100 frames only (for testing)
  python scripts/inference.py --input video.mp4 --style style.jpg --output test.mp4 \\
      --max-frames 100
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--style', '-s', type=str, required=True,
                       help='Path to style image')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output video')
    
    # Optional arguments
    parser.add_argument('--method', type=str, default='adain',
                       choices=['adain', 'johnson', 'stable_diffusion'],
                       help='Style transfer method (default: adain)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Style strength (0-1, default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (for testing)')
    parser.add_argument('--no-temporal', action='store_true',
                       help='Disable temporal consistency')
    parser.add_argument('--lightweight', action='store_true',
                       help='Use lightweight model for faster processing')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--comparison', action='store_true',
                       help='Create side-by-side comparison video')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.style):
        print(f"Error: Style image not found: {args.style}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Initialize processor
    print("Initializing video style transfer...")
    processor = VideoStyleTransfer(
        method=args.method,
        device=args.device,
        model_path=args.model_path,
        use_temporal_consistency=not args.no_temporal,
        lightweight=args.lightweight
    )
    
    # Process video
    stats = processor.process_video(
        input_path=args.input,
        style_path=args.style,
        output_path=args.output,
        alpha=args.alpha,
        max_frames=args.max_frames,
        progress=not args.no_progress
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("Processing Statistics:")
    print("="*50)
    print(f"Total frames:          {stats['total_frames']}")
    print(f"Average processing:    {stats['avg_processing_time']:.3f}s per frame")
    print(f"Average FPS:           {stats['avg_fps']:.2f}")
    print(f"Output saved to:       {stats['output_path']}")
    print("="*50)
    
    # Create comparison if requested
    if args.comparison:
        comparison_path = args.output.replace('.mp4', '_comparison.mp4')
        print(f"\nCreating comparison video: {comparison_path}")
        create_side_by_side_comparison(args.input, args.output, comparison_path)
        print("Comparison video created!")


if __name__ == "__main__":
    main()