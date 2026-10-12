"""
Experiment 5 -- JPEG Robustness Evaluation

Checkpoint: outputs/genimage_multigen5_glide_fusion.pth
Generators: SD1.5, BigGAN, VQDM, GLIDE, ADM, Wukong
JPEG conditions: Original, Q75, Q50, Q30
Threshold: 0.5
Acceptance criterion: Q50 accuracy drop <= 15 pp
"""

import os, sys, io, yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.data.multigen5_dataset import create_split_dataset
from src.models.detector import FusionDetector

BATCH_SIZE = 16
THRESHOLD = 0.5
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "genimage_multigen5_glide_fusion.pth")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
JPEG_QUALITIES = [75, 50, 30]


class JPEGDataset(Dataset):
    def __init__(self, base_dataset, quality):
        self.base_dataset = base_dataset
        self.quality = quality

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image = image.clone()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=image.dtype).view(3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=image.dtype).view(3,1,1)
        image = image * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).mul(255).byte().numpy()
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        compressed = np.array(compressed).astype(np.float32) / 255.0
        compressed = (compressed - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        compressed = torch.from_numpy(compressed).permute(2, 0, 1).float()
        return compressed, label


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)
            predictions.extend((probs >= THRESHOLD).long().cpu().numpy())
            labels.extend(targets.numpy())
    labels = np.array(labels)
    predictions = np.array(predictions)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, pos_label=0, zero_division=0),
        "recall": recall_score(labels, predictions, pos_label=0, zero_division=0),
        "f1": f1_score(labels, predictions, pos_label=0, zero_division=0),
    }


def main():
    print("\n==============================================")
    print("EXPERIMENT 5 -- JPEG ROBUSTNESS")
    print("==============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Checkpoint:\n{CHECKPOINT_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["batch_size"] = BATCH_SIZE
    config["data"]["num_workers"] = 0

    print("\nLoading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = FusionDetector(config["model"]["clip_model_name"])
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Checkpoint loaded successfully.")

    generators = {
        "SD1.5": config["data"]["sd15_path"],
        "BigGAN": config["data"]["biggan_path"],
        "VQDM": config["data"]["vqdm_path"],
        "GLIDE": config["data"]["glide_path"],
        "ADM": config["data"]["adm_path"],
        "Wukong": config["data"]["wukong_path"],
    }

    results = {}

    for gen_name, gen_path in generators.items():
        print(f"\n==============================================")
        print(f"{gen_name}")
        print(f"==============================================")
        print(f"Path: {gen_path}")

        _, _, test_dataset = create_split_dataset(gen_path, config)
        print(f"Test images: {len(test_dataset)}")

        print("\nOriginal")
        orig = evaluate(model, test_dataset, device)
        results[(gen_name, "Original")] = orig
        print(f"Accuracy:  {orig['accuracy']:.4f}")
        print(f"F1:        {orig['f1']:.4f}")

        for q in JPEG_QUALITIES:
            print(f"\nJPEG Quality {q}")
            jpeg_ds = JPEGDataset(test_dataset, q)
            m = evaluate(model, jpeg_ds, device)
            results[(gen_name, f"Q{q}")] = m
            print(f"Accuracy:  {m['accuracy']:.4f}")
            print(f"F1:        {m['f1']:.4f}")

    # Summary
    print("\n\n==============================================")
    print("JPEG ROBUSTNESS SUMMARY")
    print("==============================================")
    print("\nGenerator    Original    Q75       Q50       Q30")
    print("-" * 58)

    for gen_name in generators:
        orig = results[(gen_name, "Original")]["accuracy"]
        q75 = results[(gen_name, "Q75")]["accuracy"]
        q50 = results[(gen_name, "Q50")]["accuracy"]
        q30 = results[(gen_name, "Q30")]["accuracy"]
        drop50 = (orig - q50) * 100
        drop75 = (orig - q75) * 100
        drop30 = (orig - q30) * 100
        print(f"{gen_name:<11}{orig:.4f}     {q75:.4f}     {q50:.4f}     {q30:.4f}")
        print(f"  Q75 drop: {drop75:+.2f} pp")
        print(f"  Q50 drop: {drop50:+.2f} pp")
        print(f"  Q30 drop: {drop30:+.2f} pp")

    # Acceptance
    print("\n==============================================")
    print("ACCEPTANCE CRITERION")
    print("==============================================")
    print("JPEG Q50 accuracy drop <= 15 percentage points.\n")
    for gen_name in generators:
        orig = results[(gen_name, "Original")]["accuracy"]
        q50 = results[(gen_name, "Q50")]["accuracy"]
        drop = (orig - q50) * 100
        status = "PASS" if drop <= 15 else "FAIL"
        print(f"{gen_name:<11} {drop:>7.2f} pp  [{status}]")
    print("\n==============================================")


if __name__ == "__main__":
    main()
