import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class CLIPOnlyDetector(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        super().__init__()

        self.clip = CLIPVisionModel.from_pretrained(model_name)

        # Keep CLIP frozen, same as your current FusionDetector
        for param in self.clip.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        clip_out = self.clip(pixel_values=x).pooler_output
        return self.classifier(clip_out)