"""
Temporal Consistency Module
Ensures smooth frame-to-frame transitions using optical flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OpticalFlowEstimator(nn.Module):
    """
    Simplified optical flow estimator
    For production, use RAFT: https://github.com/princeton-vl/RAFT
    """
    
    def __init__(self):
        super(OpticalFlowEstimator, self).__init__()
        
        # Simple correlation-based flow estimation
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.flow_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, 1, 1)
        )
    
    def forward(self, frame1, frame2):
        """
        Estimate optical flow from frame1 to frame2
        
        Args:
            frame1: [N, 3, H, W]
            frame2: [N, 3, H, W]
        
        Returns:
            flow: [N, 2, H, W] - (u, v) flow vectors
        """
        feat1 = self.feature_extractor(frame1)
        feat2 = self.feature_extractor(frame2)
        
        # Simple correlation
        corr = torch.sum(feat1 * feat2, dim=1, keepdim=True)
        combined = torch.cat([feat1, feat2], dim=1)
        
        # Estimate flow
        flow = self.flow_head(feat1)
        
        # Upsample to original resolution
        flow = F.interpolate(flow, size=frame1.shape[2:], mode='bilinear', align_corners=True)
        
        return flow


def warp_frame(frame, flow):
    """
    Warp frame using optical flow
    
    Args:
        frame: [N, C, H, W] - Frame to warp
        flow: [N, 2, H, W] - Optical flow (u, v)
    
    Returns:
        warped_frame: [N, C, H, W]
    """
    N, C, H, W = frame.size()
    
    # Ensure flow matches frame dimensions
    if flow.shape[2] != H or flow.shape[3] != W:
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
    
    # Create sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=frame.device),
        torch.arange(W, device=frame.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, 2, H, W]
    
    # Add flow to grid
    flow_grid = grid + flow
    
    # Normalize to [-1, 1]
    flow_grid[:, 0] = 2.0 * flow_grid[:, 0] / (W - 1) - 1.0
    flow_grid[:, 1] = 2.0 * flow_grid[:, 1] / (H - 1) - 1.0
    
    # Reshape for grid_sample
    flow_grid = flow_grid.permute(0, 2, 3, 1)  # [N, H, W, 2]
    
    # Warp
    warped = F.grid_sample(frame, flow_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped


class TemporalConsistencyModule(nn.Module):
    """
    Module to enforce temporal consistency between consecutive frames
    """
    
    def __init__(self, flow_estimator=None, consistency_weight=1.0):
        super(TemporalConsistencyModule, self).__init__()
        
        if flow_estimator is None:
            self.flow_estimator = OpticalFlowEstimator()
        else:
            self.flow_estimator = flow_estimator
        
        self.consistency_weight = consistency_weight
        self.prev_frame = None
        self.prev_stylized = None
    
    def reset(self):
        """Reset temporal state"""
        self.prev_frame = None
        self.prev_stylized = None
    
    def forward(self, curr_frame, curr_stylized, training=False):
        """
        Apply temporal consistency
        
        Args:
            curr_frame: Current input frame [N, 3, H, W]
            curr_stylized: Current stylized frame [N, 3, H, W]
            training: Whether in training mode
        
        Returns:
            output: Temporally consistent frame
            loss: Consistency loss (if training)
        """
        if self.prev_frame is None:
            # First frame - no temporal consistency
            self.prev_frame = curr_frame.detach()
            self.prev_stylized = curr_stylized.detach()
            return curr_stylized, torch.tensor(0.0, device=curr_frame.device)
        
        # Estimate optical flow
        with torch.no_grad() if not training else torch.enable_grad():
            flow = self.flow_estimator(self.prev_frame, curr_frame)
        
        # Warp previous stylized frame
        warped_prev = warp_frame(self.prev_stylized, flow)
        
        if training:
            # Calculate consistency loss
            consistency_loss = F.l1_loss(curr_stylized, warped_prev)
            loss = self.consistency_weight * consistency_loss
        else:
            loss = torch.tensor(0.0, device=curr_frame.device)
        
        # Blend current with warped previous for smoother output
        alpha = 0.8  # Weight for current frame
        output = alpha * curr_stylized + (1 - alpha) * warped_prev
        
        # Update state
        self.prev_frame = curr_frame.detach()
        self.prev_stylized = output.detach()
        
        return output, loss
    
    def apply_temporal_smoothing(self, curr_stylized, prev_stylized, curr_frame, prev_frame):
        """
        Apply temporal smoothing without tracking internal state
        Useful for batch processing
        """
        # Estimate flow
        with torch.no_grad():
            flow = self.flow_estimator(prev_frame, curr_frame)
        
        # Warp and blend
        warped_prev = warp_frame(prev_stylized, flow)
        output = 0.8 * curr_stylized + 0.2 * warped_prev
        
        return output


class RAFTFlowEstimator:
    """
    Wrapper for RAFT optical flow estimation
    Requires: pip install git+https://github.com/princeton-vl/RAFT.git
    """
    
    def __init__(self, model_path='models/raft-things.pth'):
        try:
            import sys
            sys.path.append('external/RAFT/core')
            from raft import RAFT
            
            self.model = torch.nn.DataParallel(RAFT())
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load RAFT model: {e}")
            print("Falling back to simplified flow estimator")
            self.model = None
    
    def __call__(self, frame1, frame2):
        """Estimate flow using RAFT"""
        if self.model is None:
            # Fallback to simple estimator
            return OpticalFlowEstimator()(frame1, frame2)
        
        with torch.no_grad():
            # RAFT expects images in range [0, 255]
            frame1_scaled = frame1 * 255.0
            frame2_scaled = frame2 * 255.0
            
            # Estimate flow with multiple iterations
            flow_predictions = self.model(frame1_scaled, frame2_scaled, iters=20, test_mode=True)
            flow = flow_predictions[-1]  # Use final prediction
        
        return flow


def compute_temporal_consistency_metrics(frames, flows=None):
    """
    Compute temporal consistency metrics for a video
    
    Args:
        frames: List of frames [T, 3, H, W]
        flows: Pre-computed flows (optional)
    
    Returns:
        dict with metrics
    """
    if isinstance(frames, list):
        frames = torch.stack(frames)
    
    T, C, H, W = frames.shape
    
    # Pixel-wise temporal variance
    temporal_var = torch.var(frames, dim=0).mean().item()
    
    # Frame-to-frame difference
    frame_diffs = []
    for t in range(1, T):
        diff = F.l1_loss(frames[t], frames[t-1])
        frame_diffs.append(diff.item())
    
    avg_frame_diff = np.mean(frame_diffs)
    std_frame_diff = np.std(frame_diffs)
    
    # Temporal stability score (lower variance = more stable)
    stability_score = 1.0 / (1.0 + temporal_var)
    
    return {
        'temporal_variance': temporal_var,
        'avg_frame_difference': avg_frame_diff,
        'std_frame_difference': std_frame_diff,
        'stability_score': stability_score
    }


if __name__ == "__main__":
    # Test temporal consistency module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tcm = TemporalConsistencyModule().to(device)
    
    # Simulate video frames
    frames = [torch.randn(1, 3, 256, 256).to(device) for _ in range(10)]
    stylized_frames = [torch.randn(1, 3, 256, 256).to(device) for _ in range(10)]
    
    # Process frames
    tcm.reset()
    outputs = []
    losses = []
    
    for frame, stylized in zip(frames, stylized_frames):
        output, loss = tcm(frame, stylized, training=True)
        outputs.append(output)
        losses.append(loss.item())
    
    print(f"Processed {len(outputs)} frames")
    print(f"Average consistency loss: {np.mean(losses):.4f}")
    
    # Compute metrics
    metrics = compute_temporal_consistency_metrics(outputs)
    print("Temporal consistency metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")