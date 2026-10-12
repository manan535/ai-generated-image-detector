"""
IAD-C1 + IAD-V1 on the GenImage Stable Diffusion v1.5 subset (5K FAKE / 5K REAL).
Same architecture, same hyperparameters as the CIFAKE baseline run.
FAKE is the positive class (pos_label=0) in all reported metrics.
"""
import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch transformers security check for PyTorch < 2.6
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.data.genimage_dataset import get_genimage_dataloaders
from src.models.detector import FusionDetector


def train_and_evaluate():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    config['data']['batch_size'] = 16
    config['data']['num_workers'] = 0   # Windows: no shared-memory workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_loader, val_loader, test_loader, class_to_idx = get_genimage_dataloaders(config)
    fake_label = class_to_idx['FAKE']   # = 0
    real_label = class_to_idx['REAL']   # = 1
    print(f"\nclass_to_idx: {class_to_idx}")
    print(f"Positive class for all metrics: FAKE (label={fake_label})\n")

    model = FusionDetector(config['model']['clip_model_name']).to(device)
    trainable, frozen = model.count_parameters()
    print(f"=== Model Parameter Counts ===")
    print(f"Frozen parameters (CLIP):     {frozen:,}")
    print(f"Trainable parameters (New):   {trainable:,}")
    print(f"==============================\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')
    epochs = 10

    print(f"=== GenImage SD1.5 Subset — IAD-C1: Training ({epochs} Epochs) ===")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_time = time.time() - t0
        avg_loss   = running_loss / len(train_loader)
        peak_vram  = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0.0

        print(f"--- Epoch {epoch+1} Complete ---")
        print(f"Time: {epoch_time:.2f}s  |  Avg Loss: {avg_loss:.4f}  |  Peak VRAM: {peak_vram:.2f} MB")

    # Save checkpoint
    os.makedirs('outputs', exist_ok=True)
    ckpt = 'outputs/genimage_sd15_detector.pth'
    torch.save(model.state_dict(), ckpt)
    print(f"\nCheckpoint saved to {ckpt}")

    # === IAD-V1: Evaluation ===
    print("\n=== GenImage SD1.5 Subset — IAD-V1: Evaluation ===")
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, pos_label=fake_label, zero_division=0)
    rec  = recall_score(all_labels, all_preds,    pos_label=fake_label, zero_division=0)
    f1   = f1_score(all_labels, all_preds,        pos_label=fake_label, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds, labels=[fake_label, real_label])

    print(f"\n{'='*60}")
    print(f" GenImage SD1.5 Subset (baseline candidate)")
    print(f" Positive class = FAKE (label={fake_label})")
    print(f"{'='*60}")
    print(f"Accuracy:              {acc:.4f}")
    print(f"Precision (FAKE+):     {prec:.4f}")
    print(f"Recall    (FAKE+):     {rec:.4f}")
    print(f"F1        (FAKE+):     {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"                  Pred FAKE    Pred REAL")
    print(f"  Actual FAKE       {cm[0,0]:>6}       {cm[0,1]:>6}   (FN when wrong)")
    print(f"  Actual REAL       {cm[1,0]:>6}       {cm[1,1]:>6}   (FP when wrong)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    train_and_evaluate()
