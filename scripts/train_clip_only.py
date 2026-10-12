"""
CLIP-only baseline for GenImage Stable Diffusion v1.5 subset.

Purpose:
    Controlled comparison against the existing FusionDetector.

Architecture:
    Image
      -> CLIP ViT-B/16 frozen backbone (768-dim)
      -> classifier
      -> REAL / FAKE

Important:
    FAKE = label 0
    REAL = label 1

The sigmoid output represents P(REAL).

This experiment uses the same:
    - GenImage SD1.5 data loader
    - batch size
    - preprocessing
    - train/val/test split
    - optimizer
    - learning rate
    - number of epochs
    - evaluation threshold

as the existing FusionDetector training script.
"""

import os
import sys
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ---------------------------------------------------------------------
# Patch transformers security check for PyTorch < 2.6
# ---------------------------------------------------------------------

import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.data.genimage_dataset import get_genimage_dataloaders
from transformers import CLIPVisionModel


# ---------------------------------------------------------------------
# CLIP-only model
# ---------------------------------------------------------------------

class CLIPOnlyDetector(nn.Module):
    """
    Frozen CLIP ViT-B/16 + trainable classifier.

    CLIP ViT-B/16 pooler_output:
        768 dimensions

    Classifier:
        768 -> 256 -> 1
    """

    def __init__(self, clip_model_name):
        super().__init__()

        print(f"Loading CLIP model: {clip_model_name}")

        self.clip = CLIPVisionModel.from_pretrained(
            clip_model_name
        )

        # Freeze CLIP backbone.
        for param in self.clip.parameters():
            param.requires_grad = False

        # Same classifier dimensions/style as the FusionDetector.
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # CLIP visual representation.
        clip_out = self.clip(
            pixel_values=x
        ).pooler_output

        # Shape:
        # [batch_size, 768]

        logits = self.classifier(clip_out)

        # Shape:
        # [batch_size, 1]

        return logits

    def count_parameters(self):
        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        frozen = sum(
            p.numel()
            for p in self.parameters()
            if not p.requires_grad
        )

        return trainable, frozen


# ---------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------

def evaluate_model(model, data_loader, device, fake_label, real_label):
    """
    Evaluate using threshold = 0.5.

    IMPORTANT:
        sigmoid(logit) = P(REAL)

        Therefore:
            probability >= 0.5 -> REAL (1)
            probability <  0.5 -> FAKE (0)

    Metrics use FAKE (label 0) as the positive class.
    """

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)

            with torch.amp.autocast(
                'cuda',
                enabled=(device.type == 'cuda')
            ):
                logits = model(images)

            # sigmoid = P(REAL)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # 0 = FAKE
            # 1 = REAL
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    prec = precision_score(
        all_labels,
        all_preds,
        pos_label=fake_label,
        zero_division=0
    )

    rec = recall_score(
        all_labels,
        all_preds,
        pos_label=fake_label,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        pos_label=fake_label,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=[fake_label, real_label]
    )

    return acc, prec, rec, f1, cm


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_and_evaluate():

    # -----------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Keep EXACTLY the same settings as FusionDetector experiment.
    config['data']['batch_size'] = 16
    config['data']['num_workers'] = 0

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(
        f"Using device: {device}"
    )

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    train_loader, val_loader, test_loader, class_to_idx = \
        get_genimage_dataloaders(config)

    fake_label = class_to_idx['FAKE']
    real_label = class_to_idx['REAL']

    print(
        f"\nclass_to_idx: {class_to_idx}"
    )

    print(
        f"Positive class for all metrics: "
        f"FAKE (label={fake_label})\n"
    )

    # Safety checks.
    assert fake_label == 0, (
        f"Expected FAKE=0, got {fake_label}"
    )

    assert real_label == 1, (
        f"Expected REAL=1, got {real_label}"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    clip_model_name = config['model']['clip_model_name']

    model = CLIPOnlyDetector(
        clip_model_name
    ).to(device)

    trainable, frozen = model.count_parameters()

    print("=== Model Parameter Counts ===")

    print(
        f"Frozen parameters (CLIP):    {frozen:,}"
    )

    print(
        f"Trainable parameters (New):  {trainable:,}"
    )

    print(
        "==============================\n"
    )

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-4
    )

    # -----------------------------------------------------------------
    # AMP
    # -----------------------------------------------------------------

    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=(device.type == 'cuda')
    )

    # -----------------------------------------------------------------
    # Training configuration
    # -----------------------------------------------------------------

    epochs = 10

    print(
        f"=== GenImage SD1.5 Subset — "
        f"CLIP-only Training ({epochs} Epochs) ==="
    )

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        t0 = time.time()

        for i, (images, labels) in enumerate(
            train_loader
        ):

            images = images.to(device)

            labels = (
                labels
                .float()
                .unsqueeze(1)
                .to(device)
            )

            # Clear gradients.
            optimizer.zero_grad()

            # Forward pass.
            with torch.amp.autocast(
                'cuda',
                enabled=(device.type == 'cuda')
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            # Backpropagation.
            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            # Progress.
            if (i + 1) % 50 == 0:

                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Batch [{i + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        # -------------------------------------------------------------
        # Epoch statistics
        # -------------------------------------------------------------

        epoch_time = time.time() - t0

        avg_loss = (
            running_loss / len(train_loader)
        )

        if device.type == 'cuda':

            peak_vram = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

        else:

            peak_vram = 0.0

        print(
            f"--- Epoch {epoch + 1} Complete ---"
        )

        print(
            f"Time: {epoch_time:.2f}s  | "
            f"Avg Loss: {avg_loss:.4f}  | "
            f"Peak VRAM: {peak_vram:.2f} MB"
        )

        # -------------------------------------------------------------
        # Validation metrics
        # -------------------------------------------------------------

        (
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            _
        ) = evaluate_model(
            model,
            val_loader,
            device,
            fake_label,
            real_label
        )

        print(
            f"Validation — "
            f"Acc: {val_acc:.4f} | "
            f"Precision (FAKE): {val_prec:.4f} | "
            f"Recall (FAKE): {val_rec:.4f} | "
            f"F1 (FAKE): {val_f1:.4f}"
        )

    # -----------------------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------------------

    os.makedirs(
        'outputs',
        exist_ok=True
    )

    ckpt = (
        'outputs/'
        'genimage_sd15_clip_only.pth'
    )

    torch.save(
        model.state_dict(),
        ckpt
    )

    print(
        f"\nCheckpoint saved to {ckpt}"
    )

    # -----------------------------------------------------------------
    # IAD-V1: Final SD1.5 test evaluation
    # -----------------------------------------------------------------

    print(
        "\n=== GenImage SD1.5 Subset — "
        "CLIP-only Test Evaluation ==="
    )

    (
        acc,
        prec,
        rec,
        f1,
        cm
    ) = evaluate_model(
        model,
        test_loader,
        device,
        fake_label,
        real_label
    )

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    print(
        f"\n{'=' * 60}"
    )

    print(
        " GenImage SD1.5 Subset — CLIP-only"
    )

    print(
        f" Positive class = FAKE "
        f"(label={fake_label})"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"Accuracy:              {acc:.4f}"
    )

    print(
        f"Precision (FAKE+):     {prec:.4f}"
    )

    print(
        f"Recall    (FAKE+):     {rec:.4f}"
    )

    print(
        f"F1        (FAKE+):     {f1:.4f}"
    )

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        "                  Pred FAKE    Pred REAL"
    )

    print(
        f"  Actual FAKE       "
        f"{cm[0, 0]:>6}       "
        f"{cm[0, 1]:>6}   "
        f"(FN when wrong)"
    )

    print(
        f"  Actual REAL       "
        f"{cm[1, 0]:>6}       "
        f"{cm[1, 1]:>6}   "
        f"(FP when wrong)"
    )

    print(
        f"{'=' * 60}\n"
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == '__main__':
    train_and_evaluate()