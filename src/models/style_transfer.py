"""
Style Transfer Network using Adaptive Instance Normalization (AdaIN)
Based on: Huang & Belongie, "Arbitrary Style Transfer in Real-time" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    """VGG19-based encoder for feature extraction"""
    
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        vgg = models.vgg19(pretrained=pretrained).features
        
        # Use up to relu4_1 (first 21 layers)
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21]) # relu4_1
        
        # Freeze encoder weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Extract multi-scale features"""
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4
    
    def forward_single(self, x):
        """Extract only relu4_1 features"""
        h = self.slice1(x)
        h = self.slice2(h)
        h = self.slice3(h)
        h = self.slice4(h)
        return h


class Decoder(nn.Module):
    """Decoder network to reconstruct image from features"""
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Mirror VGG architecture
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
            nn.Sigmoid()  
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Decode features to RGB image"""
        x = self.conv4(x)
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.upsample3(x)
        x = self.conv1(x)
        return x


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for AdaIN"""
    size = feat.size()
    assert len(size) == 4, "Feature must be 4D (NCHW)"
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """
    AdaIN: Align content feature statistics to style feature statistics
    
    Args:
        content_feat: Content feature map [N, C, H, W]
        style_feat: Style feature map [N, C, H', W']
    
    Returns:
        Normalized feature map with style statistics
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    
    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean


class StyleTransferNet(nn.Module):
    """Complete AdaIN-based style transfer network"""
    
    def __init__(self, encoder=None, decoder=None):
        super(StyleTransferNet, self).__init__()
        
        if encoder is None:
            self.encoder = Encoder(pretrained=True)
        else:
            self.encoder = encoder
        
        if decoder is None:
            self.decoder = Decoder()
        else:
            self.decoder = decoder
        
        self.encoder.eval()
        # Decoder is trainable
    
    def encode(self, x):
        """Extract features"""
        return self.encoder.forward_single(x)
    
    def decode(self, t):
        """Reconstruct image from features"""
        return self.decoder(t)
    
    def forward(self, content, style, alpha=1.0):
        """Perform style transfer"""
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
    
        # Extract features
        content_feat = self.encode(content)
        style_feat = self.encode(style)
    
        # Apply AdaIN
        t = adaptive_instance_normalization(content_feat, style_feat)
    
        # Blend with original content features
        t = alpha * t + (1 - alpha) * content_feat
    
        # Decode
        output = self.decode(t)
    
        # FIX: Resize decoder output to match content dimensions
        if output.shape != content.shape:
            output = F.interpolate(output, size=content.shape[2:], mode='bilinear', align_corners=False)
    
        # Blend decoded output with original content
        output = 0.3 * output + 0.7 * content
    
        return output.clamp(0, 1)
    
    def get_features(self, x):
        """Get multi-scale features for loss computation"""
        return self.encoder(x)


class LightweightStyleTransferNet(nn.Module):
    """Lightweight version for real-time processing using MobileNetV2 backbone"""
    
    def __init__(self):
        super(LightweightStyleTransferNet, self).__init__()
        
        # Use MobileNetV2 as encoder
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.encoder = nn.Sequential(*list(mobilenet.features.children())[:14])
        
        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, content, style, alpha=1.0):
        """Simplified forward pass"""
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        
        # Apply AdaIN
        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat
        
        # Decode
        output = self.decoder(t)
        return output


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StyleTransferNet().to(device)
    
    # Create dummy inputs
    content = torch.randn(1, 3, 512, 512).to(device)
    style = torch.randn(1, 3, 512, 512).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(content, style, alpha=0.8)
    
    print(f"Input shape: {content.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
