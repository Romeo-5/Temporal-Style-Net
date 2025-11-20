"""
Interactive Gradio Demo for TemporalStyleNet
Launch with: python demo/app.py
"""

import gradio as gr
import torch
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.video_processor import VideoStyleTransfer


class StyleTransferDemo:
    """Demo application for style transfer"""
    
    def __init__(self):
        self.processor = VideoStyleTransfer(
            method='adain',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_temporal_consistency=True,
            lightweight=False  # Use lightweight model for demo
        )
        print(f"Initialized on device: {self.processor.device}")
    
    def process_video(
        self,
        video_file,
        style_image,
        alpha,
        use_temporal,
        max_frames
    ):
        """Process video with style transfer"""
        if video_file is None or style_image is None:
            return None, "Please upload both video and style image"
        
        try:
            # Update settings
            self.processor.use_temporal_consistency = use_temporal
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                output_path = tmp.name
            
            # Process video
            stats = self.processor.process_video(
                input_path=video_file,
                style_path=style_image,
                output_path=output_path,
                alpha=alpha,
                max_frames=max_frames if max_frames > 0 else None,
                progress=False
            )
            
            # Create status message
            status = f"""
            ✅ Processing Complete!
            
            📊 Statistics:
            • Total frames: {stats['total_frames']}
            • Average FPS: {stats['avg_fps']:.2f}
            • Processing time: {stats['avg_processing_time']:.3f}s per frame
            """
            
            return output_path, status
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"


def create_demo():
    """Create Gradio interface"""
    demo_app = StyleTransferDemo()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-video {
        max-height: 500px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="TemporalStyleNet Demo") as demo:
        gr.Markdown("""
        # 🎬 TemporalStyleNet Demo
        
        **Real-time video style transfer with temporal consistency**
        
        Transform your videos with artistic styles while maintaining smooth frame-to-frame transitions.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📹 Input")
                
                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"],
                )
                
                style_input = gr.Image(
                    label="Upload Style Image",
                    type="filepath",
                    sources=["upload"],
                )
                
                gr.Markdown("### ⚙️ Settings")
                
                alpha_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=1.0,
                    step=0.1,
                    label="Style Strength (α)",
                    info="Higher = more stylized"
                )
                
                temporal_checkbox = gr.Checkbox(
                    value=True,
                    label="Enable Temporal Consistency",
                    info="Smooth frame-to-frame transitions"
                )
                
                max_frames_slider = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=100,
                    step=10,
                    label="Max Frames (0 = all)",
                    info="Limit frames for faster testing"
                )
                
                process_btn = gr.Button("🎨 Process Video", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### 🎥 Output")
                
                video_output = gr.Video(
                    label="Stylized Video",
                    elem_classes=["output-video"]
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False
                )
        
        # Examples
        gr.Markdown("### 📚 Example Styles")
        gr.Markdown("""
        Try these popular artistic styles:
        - 🎨 **Starry Night** (Van Gogh) - Swirling, expressive brushstrokes
        - 🌊 **The Great Wave** (Hokusai) - Bold lines and dramatic waves
        - 🖼️ **Impression Sunrise** (Monet) - Soft, impressionist colors
        - 🎭 **The Scream** (Munch) - Intense, emotional expression
        """)
        
        # Event handlers
        process_btn.click(
            fn=demo_app.process_video,
            inputs=[
                video_input,
                style_input,
                alpha_slider,
                temporal_checkbox,
                max_frames_slider
            ],
            outputs=[video_output, status_output]
        )
        
        gr.Markdown("""
        ---
        ### 💡 Tips
        - **For faster processing**: Reduce max frames or use lighter styles
        - **For best quality**: Enable temporal consistency and use α = 1.0
        - **For subtle effects**: Reduce style strength (α < 0.7)
        
        ### 🔗 Links
        - [GitHub Repository](https://github.com/Romeo-5/temporal-style-net)
        - [Documentation](https://github.com/Romeo-5/temporal-style-net#readme)
        
        Made with ❤️ by [Romeo Nickel](https://www.linkedin.com/in/romeo-nickel/)
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Launching TemporalStyleNet Demo...")
    
    demo = create_demo()
    
    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,  # Set True to create public link
        show_error=True
    )
