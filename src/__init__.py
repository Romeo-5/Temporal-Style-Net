"""
TemporalStyleNet: Real-time video style transfer with temporal consistency
"""

__version__ = "0.1.0"
__author__ = "Romeo Nickel"

from .models.style_transfer import StyleTransferNet, LightweightStyleTransferNet
from .models.temporal_consistency import TemporalConsistencyModule
from .inference.video_processor import VideoStyleTransfer

__all__ = [
    'StyleTransferNet',
    'LightweightStyleTransferNet',
    'TemporalConsistencyModule',
    'VideoStyleTransfer',
]
