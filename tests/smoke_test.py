import unittest
import torch
import torch.nn as nn
from transformers import CLIPVisionModel
import yaml

class StubFFTBranch(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Simulated small CNN/MLP on FFT features
        self.fc = nn.Linear(3 * 224 * 224, output_dim)
        
    def forward(self, x):
        # Dummy forward pass: simulate FFT then flat feature extraction
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.fc(x_flat)

class TestEnvironmentAndModels(unittest.TestCase):
    def test_imports_and_config(self):
        # Verify PyYAML works
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.assertIn("model", config)
        self.assertEqual(config["model"]["clip_model_name"], "openai/clip-vit-base-patch16")

    def test_clip_and_fft_stub_forward(self):
        print("\nLoading CLIP ViT-B/16 vision model...")
        model_name = "openai/clip-vit-base-patch16"
        
        # Load only the vision model to save memory/download time
        vision_model = CLIPVisionModel.from_pretrained(model_name)
        vision_model.eval()
        
        # Create a random dummy image tensor (batch_size=2, channels=3, height=224, width=224)
        print("Creating dummy tensors and performing forward passes...")
        dummy_image = torch.rand(2, 3, 224, 224)
        
        # 1. Forward pass through CLIP Vision Model
        with torch.no_grad():
            outputs = vision_model(pixel_values=dummy_image)
            # pooler_output is typically what is fused
            clip_features = outputs.pooler_output 
            
        # Verify output shape from CLIP ViT-B/16 (batch_size, 768)
        self.assertEqual(clip_features.shape, (2, 768))
        print(f"CLIP Vision Model output shape verified: {clip_features.shape}")
        
        # 2. Forward pass through Stub FFT branch
        fft_stub = StubFFTBranch(output_dim=128)
        fft_features = fft_stub(dummy_image)
        
        # Verify output shape from FFT stub
        self.assertEqual(fft_features.shape, (2, 128))
        print(f"FFT Stub output shape verified: {fft_features.shape}")
        
        # 3. Dummy Fusion
        fused_features = torch.cat((clip_features, fft_features), dim=1)
        self.assertEqual(fused_features.shape, (2, 768 + 128))
        print(f"Fused feature shape verified: {fused_features.shape}")

if __name__ == "__main__":
    unittest.main()
