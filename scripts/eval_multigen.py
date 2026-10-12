import os
import sys
import yaml
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from src.data.multigen_dataset import create_split_dataset
from src.models.detector import FusionDetector


CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "config.yaml"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "genimage_multigen_fusion.pth"
)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def evaluate_dataset(
    model,
    dataset,
    device,
    name,
    batch_size=16,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            # Model output is P(REAL)
            probs = torch.sigmoid(
                logits
            ).squeeze(1)

            # 0 = FAKE
            # 1 = REAL
            preds = (
                probs >= 0.5
            ).long()

            all_labels.extend(
                labels.numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        pos_label=0,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=[0, 1]
    )

    fake_probs = all_probs[
        all_labels == 0
    ]

    real_probs = all_probs[
        all_labels == 1
    ]

    print("\n===================================")
    print(name)
    print("===================================")

    print(
        f"Samples: {len(all_labels)}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print("\nConfusion Matrix")
    print(
        "              Pred FAKE   Pred REAL"
    )
    print(
        f"Actual FAKE   {cm[0,0]:10d}   {cm[0,1]:10d}"
    )
    print(
        f"Actual REAL   {cm[1,0]:10d}   {cm[1,1]:10d}"
    )

    print("\nP(REAL) distribution")

    print(
        f"FAKE mean: {fake_probs.mean():.4f}"
    )

    print(
        f"FAKE std:  {fake_probs.std():.4f}"
    )

    print(
        f"REAL mean: {real_probs.mean():.4f}"
    )

    print(
        f"REAL std:  {real_probs.std():.4f}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    config = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    print("\nLoading checkpoint:")
    print(CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model = FusionDetector(
        config["model"]["clip_model_name"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print("\nCheckpoint loaded successfully.")

    print(
        "\nBest validation F1 from training:",
        f"{checkpoint.get('val_metrics', {}).get('f1', 0):.4f}"
    )

    # --------------------------------------------------
    # Generator paths
    # --------------------------------------------------

    generators = {
        "SD1.5 TEST": config["data"]["sd15_path"],
        "BigGAN TEST": config["data"]["biggan_path"],
        "ADM TEST": "D:/IAD_data/genimage_adm",
    }

    results = {}

    for name, path in generators.items():

        print("\n-----------------------------------")
        print(f"Loading {name}")
        print(path)
        print("-----------------------------------")

        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Generator directory not found:\n{path}"
            )

        # create_split_dataset uses the exact same
        # deterministic 70/15/15 split used during training.
        _, _, test_dataset = create_split_dataset(
            path,
            config
        )

        results[name] = evaluate_dataset(
            model,
            test_dataset,
            device,
            name,
            batch_size=config["data"]["batch_size"]
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n\n===================================")
    print("MULTI-GENERATOR EVALUATION SUMMARY")
    print("===================================")

    print(
        "\nGenerator       Accuracy   Precision   Recall     F1"
    )

    for name, metrics in results.items():

        print(
            f"{name:<15}"
            f"{metrics['accuracy']:.4f}     "
            f"{metrics['precision']:.4f}      "
            f"{metrics['recall']:.4f}      "
            f"{metrics['f1']:.4f}"
        )

    print("\nPrimary result:")
    print(
        "ADM = completely unseen generator during training."
    )


if __name__ == "__main__":
    main()