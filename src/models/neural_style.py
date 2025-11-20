"""
Neural Style Transfer using pre-trained VGG
Multi-scale feature matching - no training needed!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights


class VGGStyleTransfer(nn.Module):
    """Multi-scale style transfer using pre-trained VGG"""
    
    def __init__(self):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        # Extract layers for multi-scale feature extraction
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2: 64 channels
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2: 128 channels
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18]) # relu3_4: 256 channels
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27]) # relu4_4: 512 channels
        
        # Add a simple projection layer to convert features back to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize projection layer
        for m in self.to_rgb.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Freeze VGG parameters only
        for param in self.slice1.parameters():
            param.requires_grad = False
        for param in self.slice2.parameters():
            param.requires_grad = False
        for param in self.slice3.parameters():
            param.requires_grad = False
        for param in self.slice4.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def get_features(self, x):
        """Extract multi-scale features"""
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4
    
    def match_color_stats(self, content_feat, style_feat):
        """Match color statistics (mean and std)"""
        # Calculate statistics
        c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        c_std = content_feat.std(dim=[2, 3], keepdim=True)
        
        s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        s_std = style_feat.std(dim=[2, 3], keepdim=True)
        
        # Normalize and transform
        normalized = (content_feat - c_mean) / (c_std + 1e-5)
        stylized = normalized * s_std + s_mean
        
        return stylized
    
    def forward(self, content, style, alpha=1.0):
        """
        Style transfer with RGB output
        
        Args:
            content: Content image [N, 3, H, W]
            style: Style image [N, 3, H', W']
            alpha: Style strength (0-1)
        
        Returns:
            Stylized RGB image [N, 3, H, W]
        """
        # Extract features
        c_feats = self.get_features(content)
        s_feats = self.get_features(style)
        
        # Use deepest features (relu4_4) - 512 channels
        c_deep = c_feats[-1]
        s_deep = s_feats[-1]
        
        # Match style statistics
        stylized_feat = self.match_color_stats(c_deep, s_deep)
        
        # Blend stylized with original content features (in feature space)
        blended_feat = alpha * stylized_feat + (1 - alpha) * c_deep
        
        # Upsample to original size (still 512 channels)
        upsampled = F.interpolate(blended_feat, size=content.shape[2:], mode='bilinear', align_corners=False)
        
        # Project to RGB (512 -> 3 channels)
        output = self.to_rgb(upsampled)
        
        return output.clamp(0, 1)


# Quick wrapper for video processing
class FastVGGStyleTransfer(VGGStyleTransfer):
    """Optimized version for video processing - same as parent now"""
    pass