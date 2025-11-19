"""
Video Style Transfer Pipeline
Complete inference pipeline for processing videos with style transfer
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio
from torchvision import transforms

from ..models.style_transfer import StyleTransferNet, LightweightStyleTransferNet
from ..models.temporal_consistency import TemporalConsistencyModule


class VideoStyleTransfer:
    """Complete video style transfer pipeline"""
    
    def __init__(
        self,
        method='adain',
        device='cuda',
        model_path=None,
        use_temporal_consistency=True,
        lightweight=False
    ):
        """
        Initialize video style transfer processor
        
        Args:
            method: 'adain', 'johnson', or 'stable_diffusion'
            device: 'cuda' or 'cpu'
            model_path: Path to pre-trained model (optional)
            use_temporal_consistency: Enable temporal consistency
            lightweight: Use lightweight model for speed
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.use_temporal_consistency = use_temporal_consistency
        
        # Initialize models
        if lightweight:
            self.style_net = LightweightStyleTransferNet().to(self.device)
        else:
            self.style_net = StyleTransferNet().to(self.device)
        
        if model_path is not None:
            self.load_model(model_path)
        
        self.style_net.eval()
        
        # Temporal consistency module
        if use_temporal_consistency:
            self.temporal_module = TemporalConsistencyModule().to(self.device)
        else:
            self.temporal_module = None
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.denorm = transforms.Compose([
            transforms.Lambda(lambda x: x.clamp(0, 1)),
        ])
    
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'decoder' in checkpoint:
            self.style_net.decoder.load_state_dict(checkpoint['decoder'])
        else:
            self.style_net.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
    
    def load_image(self, image_path, size=None):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        
        if size is not None:
            image = image.resize((size, size), Image.LANCZOS)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def load_video(self, video_path):
        """Load video and extract frames"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        return frames, fps, (width, height)
    
    def save_video(self, frames, output_path, fps=30):
        """Save frames as video"""
        print(f"Saving video to {output_path}")
        
        # Convert frames to uint8
        frames_uint8 = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            frames_uint8.append(frame)
        
        # Save with imageio
        imageio.mimsave(output_path, frames_uint8, fps=fps)
        print(f"Video saved: {output_path}")
    
    def process_frame(self, frame, style_image, alpha=1.0):
        """
        Process single frame with style transfer
        
        Args:
            frame: numpy array [H, W, 3] or tensor [3, H, W]
            style_image: tensor [1, 3, H, W]
            alpha: style strength
        
        Returns:
            Stylized frame as numpy array [H, W, 3]
        """
        # Convert to tensor if needed
        if isinstance(frame, np.ndarray):
            frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        else:
            frame_tensor = frame.unsqueeze(0).to(self.device)
        
        # Apply style transfer
        with torch.no_grad():
            stylized = self.style_net(frame_tensor, style_image, alpha=alpha)
        
        # Convert back to numpy
        stylized = self.denorm(stylized.squeeze(0)).cpu().numpy()
        stylized = np.transpose(stylized, (1, 2, 0))
        
        return stylized
    
    def process_video(
        self,
        input_path,
        style_path,
        output_path,
        alpha=1.0,
        max_frames=None,
        progress=True
    ):
        """
        Process entire video with style transfer
        
        Args:
            input_path: Path to input video
            style_path: Path to style image
            output_path: Path to save output video
            alpha: Style strength (0-1)
            max_frames: Maximum number of frames to process (for testing)
            progress: Show progress bar
        
        Returns:
            dict with processing statistics
        """
        print(f"Processing video: {input_path}")
        print(f"Style: {style_path}")
        
        # Load inputs
        frames, fps, (width, height) = self.load_video(input_path)
        style_image = self.load_image(style_path)
        
        if max_frames is not None:
            frames = frames[:max_frames]
        
        # Reset temporal state
        if self.temporal_module is not None:
            self.temporal_module.reset()
        
        # Process frames
        stylized_frames = []
        processing_times = []
        
        iterator = tqdm(frames, desc="Processing frames") if progress else frames
        
        import time
        for i, frame in enumerate(iterator):
            start_time = time.time()
            
            # Convert frame to tensor
            frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            
            # Apply style transfer
            with torch.no_grad():
                stylized = self.style_net(frame_tensor, style_image, alpha=alpha)
            
            # Apply temporal consistency
            if self.temporal_module is not None and i > 0:
                stylized, _ = self.temporal_module(frame_tensor, stylized, training=False)
            
            # Convert to numpy
            stylized = self.denorm(stylized.squeeze(0)).cpu()
            stylized = stylized.permute(1, 2, 0).numpy()
            stylized = (stylized * 255).astype(np.uint8)
            
            stylized_frames.append(stylized)
            
            elapsed = time.time() - start_time
            processing_times.append(elapsed)
        
        # Save output
        self.save_video(stylized_frames, output_path, fps=fps)
        
        # Compute statistics
        avg_time = np.mean(processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        stats = {
            'total_frames': len(frames),
            'avg_processing_time': avg_time,
            'avg_fps': avg_fps,
            'output_path': output_path
        }
        
        print(f"\nProcessing complete!")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total time: {sum(processing_times):.2f}s")
        
        return stats
    
    def process_batch(
        self,
        input_paths,
        style_paths,
        output_dir,
        alpha=1.0
    ):
        """Process multiple videos"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, (video_path, style_path) in enumerate(zip(input_paths, style_paths)):
            output_path = os.path.join(
                output_dir,
                f"stylized_{i}_{os.path.basename(video_path)}"
            )
            
            stats = self.process_video(
                video_path,
                style_path,
                output_path,
                alpha=alpha
            )
            
            results.append(stats)
        
        return results
    
    def compare_methods(self, input_path, style_path, output_dir):
        """Compare different style transfer methods"""
        os.makedirs(output_dir, exist_ok=True)
        
        methods = ['adain']  # Add more as implemented
        results = {}
        
        for method in methods:
            self.method = method
            output_path = os.path.join(output_dir, f"{method}_output.mp4")
            
            stats = self.process_video(
                input_path,
                style_path,
                output_path
            )
            
            results[method] = stats
        
        return results


def create_side_by_side_comparison(original_path, stylized_path, output_path):
    """Create side-by-side comparison video"""
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(stylized_path)
    
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Concatenate horizontally
        combined = np.hstack([frame1, frame2])
        out.write(combined)
    
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"Comparison video saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    processor = VideoStyleTransfer(
        method='adain',
        device='cuda',
        use_temporal_consistency=True
    )
    
    # Process single video
    stats = processor.process_video(
        input_path='data/videos/input.mp4',
        style_path='data/styles/starry_night.jpg',
        output_path='data/outputs/stylized.mp4',
        alpha=0.8
    )
    
    print("Processing statistics:", stats)
