"""
eval_multigen5.py — Experiment 5 Evaluation

Evaluates on: SD1.5, BigGAN, VQDM, GLIDE (seen), ADM, Wukong (held-out)
"""

import os
import sys
import yaml
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from src.data.multigen5_dataset import get_multigen5_dataloaders


def evaluate(model, loader, device, fake_label, real_label, label=""):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels_batch.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, pos_label=fake_label, zero_division=0)
    rec = recall_score(all_labels, all_preds, pos_label=fake_label, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=fake_label, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[fake_label, real_label])

    fake_probs = all_probs[all_labels == fake_label]
    real_probs = all_probs[all_labels == real_label]

    print(f"\n{'=' * 60}")
    print(f" {label}")
    print(f" Positive class = FAKE (label={fake_label})")
    print(f"{'=' * 60}")
    print(f"Accuracy:              {acc:.4f}")
    print(f"Precision (FAKE+):     {prec:.4f}")
    print(f"Recall    (FAKE+):     {rec:.4f}")
    print(f"F1        (FAKE+):     {f1:.4f}")
    print(f"FN (Missed Fakes):     {cm[0, 1]}")
    print(f"FP (False Reals):      {cm[1, 0]}")
    print("\nConfusion Matrix:")
    print("                  Pred FAKE    Pred REAL")
    print(f"  Actual FAKE       {cm[0,0]:>6}       {cm[0,1]:>6}   (FN when wrong)")
    print(f"  Actual REAL       {cm[1,0]:>6}       {cm[1,1]:>6}   (FP when wrong)")

    if len(fake_probs) > 0:
        print(f"\nP(REAL) for FAKE images:  Mean = {np.mean(fake_probs):.4f}, Std = {np.std(fake_probs):.4f}")
    if len(real_probs) > 0:
        print(f"P(REAL) for REAL images:  Mean = {np.mean(real_probs):.4f}, Std = {np.std(real_probs):.4f}")
    print(f"{'=' * 60}")


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    config['data']['batch_size'] = 16
    config['data']['num_workers'] = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _, _, test_datasets = get_multigen5_dataloaders(config)

    model = FusionDetector(config['model']['clip_model_name']).to(device)

    ckpt = 'outputs/genimage_multigen5_glide_fusion.pth'
    print(f"\nLoading weights from {ckpt}")

    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    for generator_name, dataset in test_datasets.items():
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config['data']['batch_size'],
            shuffle=False, num_workers=0
        )
        evaluate(
            model, loader, device,
            fake_label=0, real_label=1,
            label=f"TEST SET: {generator_name.upper()}"
        )


if __name__ == '__main__':
    main()
