import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import CLIPVisionModel

class FrequencyBranch(nn.Module):
    """
    2D-FFT -> log-magnitude spectrum -> small CNN -> 128-dim embedding.
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        # Convert RGB to grayscale (shape: [B, 1, 224, 224])
        gray = TF.rgb_to_grayscale(x)
        
        # 2D FFT
        fft = torch.fft.fft2(gray, norm='ortho')
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft_shift)
        log_mag = torch.log(mag + 1e-8)
        
        # CNN Feature Extraction
        feat = self.features(log_mag)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out

class FusionDetector(nn.Module):
    """
    Combines frozen CLIP backbone with trainable Frequency Branch.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        # Load vision tower only
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        
        # Freeze the entire CLIP backbone (IAD-C1)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.freq_branch = FrequencyBranch(out_dim=128)
        
        # Classifier Head (768 from CLIP + 128 from FFT = 896)
        self.classifier = nn.Sequential(
            nn.Linear(896, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary classification logits
        )

    def forward(self, x):
        # Forward pass through frozen CLIP
        clip_out = self.clip(pixel_values=x).pooler_output
        
        # Forward pass through trainable frequency branch
        freq_out = self.freq_branch(x)
        
        # Fusion
        fused = torch.cat([clip_out, freq_out], dim=1)
        
        # Classifier logits
        logits = self.classifier(fused)
        return logits
    
    def count_parameters(self):
        """Returns tuple of (trainable_params, frozen_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return trainable, frozen
