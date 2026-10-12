"""
Evaluate an existing checkpoint with corrected FAKE-as-positive-class metrics.

class_to_idx confirmed: {'FAKE': 0, 'REAL': 1}
Therefore pos_label=0 for all precision/recall/F1 calls — FAKE is the positive class
for a fake detector, not REAL.
"""
import os
import sys
import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch transformers security check for PyTorch < 2.6
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.data.dataset import get_dataloaders
from src.models.detector import FusionDetector


def evaluate(checkpoint_path: str, config: dict, dataset_label: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    config['data']['num_workers'] = 0  # Windows: no shared memory workers

    _, _, test_loader = get_dataloaders(config)

    # Confirm class mapping
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(os.path.join(config['data']['cifake_path'], 'train'))
    print(f"\nclass_to_idx: {ds.class_to_idx}")
    fake_label = ds.class_to_idx['FAKE']   # confirmed = 0
    real_label = ds.class_to_idx['REAL']   # confirmed = 1
    print(f"FAKE label index: {fake_label}  (pos_label for all metrics)")
    print(f"REAL label index: {real_label}")

    model = FusionDetector(config['model']['clip_model_name']).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"\nLoaded checkpoint: {checkpoint_path}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            # Model outputs high value → REAL (label 1), low value → FAKE (label 0)
            preds = (probs >= 0.5).astype(int).flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    # --- Metrics with FAKE (label=0) as the positive class ---
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, pos_label=fake_label, zero_division=0)
    rec  = recall_score(all_labels, all_preds,    pos_label=fake_label, zero_division=0)
    f1   = f1_score(all_labels, all_preds,        pos_label=fake_label, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds, labels=[fake_label, real_label])

    print(f"\n{'='*60}")
    print(f" {dataset_label}")
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
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    evaluate(
        checkpoint_path='outputs/baseline_detector.pth',
        config=config,
        dataset_label='CIFAKE (smoke test, corrected metrics — FAKE is positive class)'
    )
