"""
Evaluate the CLIP-only GenImage detector.

Checkpoint:
    outputs/genimage_sd15_clip_only.pth

Evaluates:
    1. SD1.5  - in-distribution
    2. BigGAN - unseen generator
    3. ADM    - unseen generator

Label convention:
    FAKE = 0
    REAL = 1

Therefore:
    sigmoid(logit) = P(REAL)

    P(REAL) >= 0.5 -> REAL
    P(REAL) <  0.5 -> FAKE
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =====================================================================
# PROJECT ROOT
# =====================================================================

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


# =====================================================================
# TRANSFORMERS COMPATIBILITY PATCH
# =====================================================================

import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = (
    lambda: None
)

transformers.modeling_utils.check_torch_load_is_safe = (
    lambda: None
)


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from src.data.genimage_dataset import get_genimage_dataloaders
from transformers import CLIPVisionModel


# =====================================================================
# CLIP-ONLY MODEL
# =====================================================================

class CLIPOnlyDetector(nn.Module):

    def __init__(self, clip_model_name):

        super().__init__()

        print(
            f"Loading CLIP model: {clip_model_name}"
        )

        self.clip = CLIPVisionModel.from_pretrained(
            clip_model_name
        )

        # -------------------------------------------------------------
        # Freeze CLIP backbone
        # -------------------------------------------------------------

        for param in self.clip.parameters():
            param.requires_grad = False

        # -------------------------------------------------------------
        # Same classifier used during training
        # -------------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):

        clip_out = self.clip(
            pixel_values=x
        ).pooler_output

        # clip_out shape:
        # [batch_size, 768]

        logits = self.classifier(
            clip_out
        )

        # logits shape:
        # [batch_size, 1]

        return logits


# =====================================================================
# EVALUATE ONE DATASET
# =====================================================================

def evaluate_dataset(
    model,
    loader,
    device,
    dataset_name,
    fake_label,
    real_label
):

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda")
            ):

                logits = model(images)

            # ---------------------------------------------------------
            # IMPORTANT:
            #
            # sigmoid(logit) = P(REAL)
            #
            # because:
            # FAKE = 0
            # REAL = 1
            # ---------------------------------------------------------

            probs = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
                .flatten()
            )

            # ---------------------------------------------------------
            # Threshold = 0.5
            #
            # P(REAL) >= 0.5 -> REAL = 1
            # P(REAL) <  0.5 -> FAKE = 0
            # ---------------------------------------------------------

            preds = (
                probs >= 0.5
            ).astype(int)

            all_probs.extend(
                probs.tolist()
            )

            all_preds.extend(
                preds.tolist()
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

    # -------------------------------------------------------------
    # Convert to numpy
    # -------------------------------------------------------------

    all_probs = np.array(
        all_probs
    )

    all_preds = np.array(
        all_preds
    )

    all_labels = np.array(
        all_labels
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        pos_label=fake_label,
        zero_division=0
    )

    recall = recall_score(
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
        labels=[
            fake_label,
            real_label
        ]
    )

    # -------------------------------------------------------------
    # Probability distributions
    # -------------------------------------------------------------

    fake_probs = all_probs[
        all_labels == fake_label
    ]

    real_probs = all_probs[
        all_labels == real_label
    ]

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print(f" {dataset_name}")
    print("=" * 70)

    print(
        f"Accuracy:              {acc:.4f}"
    )

    print(
        f"Precision (FAKE+):     {precision:.4f}"
    )

    print(
        f"Recall    (FAKE+):     {recall:.4f}"
    )

    print(
        f"F1        (FAKE+):     {f1:.4f}"
    )

    print()

    # -------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------

    print(
        "Confusion Matrix:"
    )

    print(
        "                  Pred FAKE    Pred REAL"
    )

    print(
        f"Actual FAKE       "
        f"{cm[0, 0]:>6}       "
        f"{cm[0, 1]:>6}"
    )

    print(
        f"Actual REAL       "
        f"{cm[1, 0]:>6}       "
        f"{cm[1, 1]:>6}"
    )

    print()

    # -------------------------------------------------------------
    # Probability distribution
    # -------------------------------------------------------------

    print(
        "P(REAL) distribution:"
    )

    if len(fake_probs) > 0:

        print(
            f"FAKE images: "
            f"mean={fake_probs.mean():.4f}, "
            f"std={fake_probs.std():.4f}, "
            f"min={fake_probs.min():.4f}, "
            f"max={fake_probs.max():.4f}"
        )

    if len(real_probs) > 0:

        print(
            f"REAL images: "
            f"mean={real_probs.mean():.4f}, "
            f"std={real_probs.std():.4f}, "
            f"min={real_probs.min():.4f}, "
            f"max={real_probs.max():.4f}"
        )

    print(
        "=" * 70
    )

    return {
        "dataset": dataset_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
    }


# =====================================================================
# LOAD TEST DATASET
# =====================================================================

def get_test_loader(
    base_config,
    generator_path
):

    # -------------------------------------------------------------
    # Make a fresh config
    # -------------------------------------------------------------

    config = yaml.safe_load(
        open("config.yaml")
    )

    # -------------------------------------------------------------
    # Set the parameters explicitly
    # -------------------------------------------------------------

    config["data"]["batch_size"] = 16

    config["data"]["num_workers"] = 0

    # THIS IS THE IMPORTANT FIX.
    #
    # genimage_dataset.py reads:
    #
    # config["data"]["genimage_path"]
    #
    # NOT:
    #
    # config["data"]["root"]
    # -------------------------------------------------------------

    config["data"]["genimage_path"] = generator_path

    # -------------------------------------------------------------
    # Print path before loader
    # -------------------------------------------------------------

    print()
    print(
        "--------------------------------------------------"
    )

    print(
        f"VERIFY PATH:"
    )

    print(
        f"{config['data']['genimage_path']}"
    )

    print(
        "--------------------------------------------------"
    )

    # -------------------------------------------------------------
    # Build loader
    # -------------------------------------------------------------

    _, _, test_loader, class_to_idx = \
        get_genimage_dataloaders(
            config
        )

    # -------------------------------------------------------------
    # Verify class mapping
    # -------------------------------------------------------------

    print(
        f"class_to_idx: {class_to_idx}"
    )

    return (
        test_loader,
        class_to_idx
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # =================================================================
    # LOAD CONFIG
    # =================================================================

    with open("config.yaml") as f:

        config = yaml.safe_load(
            f
        )

    # =================================================================
    # DEVICE
    # =================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        torch.cuda.empty_cache()

    # =================================================================
    # LOAD MODEL
    # =================================================================

    clip_model_name = (
        config["model"]["clip_model_name"]
    )

    model = CLIPOnlyDetector(
        clip_model_name
    ).to(device)

    # =================================================================
    # LOAD CHECKPOINT
    # =================================================================

    checkpoint = (
        "outputs/"
        "genimage_sd15_clip_only.pth"
    )

    print()
    print(
        f"Loading checkpoint:"
    )

    print(
        checkpoint
    )

    if not os.path.exists(checkpoint):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{checkpoint}\n\n"
            f"Make sure the CLIP-only model was trained "
            f"and saved to outputs/genimage_sd15_clip_only.pth"
        )

    state_dict = torch.load(
        checkpoint,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    print(
        "Checkpoint loaded successfully."
    )

    # =================================================================
    # DATASET PATHS
    # =================================================================

    datasets = [

        (
            "SD1.5 — In-Distribution",
            r"D:\IAD_data\genimage_sd15"
        ),

        (
            "BigGAN — Unseen Generator",
            r"D:\IAD_data\genimage_biggan"
        ),

        (
            "ADM — Unseen Generator",
            r"D:\IAD_data\genimage_adm"
        ),

    ]

    # =================================================================
    # RESULTS
    # =================================================================

    results = []

    # =================================================================
    # EVALUATE EACH DATASET
    # =================================================================

    for dataset_name, dataset_path in datasets:

        # -------------------------------------------------------------
        # Make sure folder exists
        # -------------------------------------------------------------

        if not os.path.exists(dataset_path):

            raise FileNotFoundError(
                f"\nDataset folder not found:\n"
                f"{dataset_path}"
            )

        # -------------------------------------------------------------
        # Make sure FAKE and REAL exist
        # -------------------------------------------------------------

        fake_dir = os.path.join(
            dataset_path,
            "FAKE"
        )

        real_dir = os.path.join(
            dataset_path,
            "REAL"
        )

        if not os.path.exists(fake_dir):

            raise FileNotFoundError(
                f"\nFAKE directory missing:\n"
                f"{fake_dir}"
            )

        if not os.path.exists(real_dir):

            raise FileNotFoundError(
                f"\nREAL directory missing:\n"
                f"{real_dir}"
            )

        # -------------------------------------------------------------
        # Load dataset
        # -------------------------------------------------------------

        test_loader, class_to_idx = \
            get_test_loader(
                config,
                dataset_path
            )

        fake_label = class_to_idx["FAKE"]

        real_label = class_to_idx["REAL"]

        # -------------------------------------------------------------
        # Safety checks
        # -------------------------------------------------------------

        if fake_label != 0:

            raise ValueError(
                f"Expected FAKE=0 but got "
                f"FAKE={fake_label}"
            )

        if real_label != 1:

            raise ValueError(
                f"Expected REAL=1 but got "
                f"REAL={real_label}"
            )

        # -------------------------------------------------------------
        # Evaluate
        # -------------------------------------------------------------

        result = evaluate_dataset(
            model=model,
            loader=test_loader,
            device=device,
            dataset_name=dataset_name,
            fake_label=fake_label,
            real_label=real_label,
        )

        results.append(
            result
        )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print()
    print("=" * 95)
    print(
        " CLIP-ONLY CROSS-GENERATOR SUMMARY"
    )
    print("=" * 95)

    print(
        f"{'Dataset':<35}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print(
        "-" * 95
    )

    for result in results:

        print(
            f"{result['dataset']:<35}"
            f"{result['accuracy']:>12.4f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
        )

    print(
        "=" * 95
    )

    # =================================================================
    # FUSION REFERENCE
    # =================================================================

    print()
    print(
        "Existing FusionDetector reference:"
    )

    print(
        "SD1.5  : Accuracy=0.9250 | "
        "Precision=0.8806 | "
        "Recall=0.9833 | "
        "F1=0.9291"
    )

    print(
        "BigGAN : Accuracy=0.4833 | "
        "Precision=0.4444 | "
        "Recall=0.1333 | "
        "F1=0.2051"
    )

    print(
        "ADM    : Accuracy=0.4900 | "
        "Precision=0.4667 | "
        "Recall=0.1400 | "
        "F1=0.2154"
    )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()