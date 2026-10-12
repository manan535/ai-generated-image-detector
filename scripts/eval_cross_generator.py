import os
import sys

import yaml
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

# Disable transformers torch-load safety checks
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.data.genimage_dataset import get_genimage_dataloaders
from src.models.detector import FusionDetector


def evaluate_threshold(labels, probs, threshold, fake_label, real_label):
    """
    Evaluate detector performance at a given threshold.

    FAKE is treated as the positive class.
    """

    # Convert probabilities to predicted labels
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)

    prec = precision_score(
        labels,
        preds,
        pos_label=fake_label,
        zero_division=0,
    )

    rec = recall_score(
        labels,
        preds,
        pos_label=fake_label,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        preds,
        pos_label=fake_label,
        zero_division=0,
    )

    cm = confusion_matrix(
        labels,
        preds,
        labels=[fake_label, real_label],
    )

    return acc, prec, rec, f1, cm


def main():

    # Generator can be passed from command line.
    # Example:
    # python eval_cross_generator.py adm
    generator = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "biggan"
    )

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Keep evaluation settings deterministic and Windows-friendly
    config["data"]["batch_size"] = 16
    config["data"]["num_workers"] = 0

    # Generator-specific GenImage directory
    config["data"]["genimage_path"] = (
        f"D:/IAD_data/genimage_{generator}"
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Evaluating SD1.5 model on "
        f"Unseen Generator: {generator.upper()}"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Load GenImage test split
    # ---------------------------------------------------------

    # The test loader contains:
    # 300 REAL + 300 FAKE
    _, _, test_loader, class_to_idx = (
        get_genimage_dataloaders(config)
    )

    fake_label = class_to_idx["FAKE"]
    real_label = class_to_idx["REAL"]

    print(f"FAKE label: {fake_label}")
    print(f"REAL label: {real_label}")

    # ---------------------------------------------------------
    # Load trained SD1.5 checkpoint
    # ---------------------------------------------------------

    ckpt = "outputs/genimage_sd15_detector.pth"

    model = FusionDetector(
        config["model"]["clip_model_name"]
    ).to(device)

    model.load_state_dict(
        torch.load(
            ckpt,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    print(f"Checkpoint: {ckpt}")
    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # Collect probabilities
    # ---------------------------------------------------------

    print(
        "\nCollecting probabilities "
        "on cross-generator test split..."
    )

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            # Use autocast only when CUDA is available
            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            # Convert logits to sigmoid probabilities
            probs = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
                .flatten()
            )

            all_probs.extend(
                probs.tolist()
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

    # ---------------------------------------------------------
    # Convert to NumPy arrays
    # ---------------------------------------------------------

    val_labels = np.array(all_labels)
    val_probs = np.array(all_probs)

    # ---------------------------------------------------------
    # Raw sigmoid distribution for FAKE images
    # ---------------------------------------------------------

    fake_probs = val_probs[
        val_labels == fake_label
    ]

    print(
        f"\n{generator.upper()} fake images "
        f"— raw sigmoid distribution:"
    )

    print(
        f"  mean={fake_probs.mean():.3f}"
    )

    print(
        f"  std={fake_probs.std():.3f}"
    )

    print(
        f"  min={fake_probs.min():.3f}"
    )

    print(
        f"  max={fake_probs.max():.3f}"
    )

    print(
        "  25th/50th/75th percentile:",
        np.percentile(
            fake_probs,
            [25, 50, 75],
        ),
    )

    # ---------------------------------------------------------
    # Evaluate thresholds
    # ---------------------------------------------------------

    # 0.5 = primary threshold
    # 0.0837 = exploratory threshold tuned on SD1.5
    thresholds = [
        0.5,
        0.0837,
    ]

    print(
        f"\n{'=' * 100}"
    )

    print(
        f"{'Threshold':<10} | "
        f"{'Prec':<8} | "
        f"{'Rec':<8} | "
        f"{'F1':<8} | "
        f"{'Acc':<8} | "
        f"{'FN (Missed)':<15} | "
        f"{'FP (False Accusation)':<25}"
    )

    print(
        f"{'-' * 100}"
    )

    for threshold in thresholds:

        acc, prec, rec, f1, cm = evaluate_threshold(
            val_labels,
            val_probs,
            threshold,
            fake_label,
            real_label,
        )

        # Confusion matrix layout:
        #
        #              Pred FAKE   Pred REAL
        # Actual FAKE      TP          FN
        # Actual REAL      FP          TN
        #
        # Because labels=[fake_label, real_label]:
        #
        # cm[0, 1] = FN
        # cm[1, 0] = FP

        fn = cm[0, 1]
        fp = cm[1, 0]

        print(
            f"{threshold:<10.4f} | "
            f"{prec:<8.4f} | "
            f"{rec:<8.4f} | "
            f"{f1:<8.4f} | "
            f"{acc:<8.4f} | "
            f"{fn:<15} | "
            f"{fp:<25}"
        )

    print(
        f"{'=' * 100}\n"
    )


if __name__ == "__main__":
    main()