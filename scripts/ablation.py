import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# TRANSFORMERS COMPATIBILITY WORKAROUND
# =========================================================

import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = (
    lambda: None
)

transformers.modeling_utils.check_torch_load_is_safe = (
    lambda: None
)


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.data.genimage_dataset import (
    get_genimage_dataloaders,
)

from src.models.detector import FusionDetector


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "genimage_sd15_detector.pth",
)

BATCH_SIZE = 16
NUM_WORKERS = 0

THRESHOLD = 0.5


# =========================================================
# MODEL WRAPPER
# =========================================================

class AblationDetector(nn.Module):
    """
    Runs the existing FusionDetector while allowing
    one branch to be disabled.

    Modes:

        full
            CLIP + frequency

        clip_only
            CLIP + zero frequency

        frequency_only
            zero CLIP + frequency
    """

    def __init__(
        self,
        base_model,
        mode="full",
    ):
        super().__init__()

        self.model = base_model
        self.mode = mode

        if mode not in {
            "full",
            "clip_only",
            "frequency_only",
        }:
            raise ValueError(
                f"Unknown ablation mode: {mode}"
            )

    def forward(self, x):

        # -------------------------------------------------
        # CLIP branch
        # -------------------------------------------------

        clip_out = self.model.clip(
            pixel_values=x
        ).pooler_output

        # -------------------------------------------------
        # Frequency branch
        # -------------------------------------------------

        freq_out = self.model.freq_branch(x)

        # -------------------------------------------------
        # Apply ablation
        # -------------------------------------------------

        if self.mode == "clip_only":

            # Keep CLIP.
            # Remove frequency contribution.

            freq_out = torch.zeros_like(
                freq_out
            )

        elif self.mode == "frequency_only":

            # Keep frequency.
            # Remove CLIP contribution.

            clip_out = torch.zeros_like(
                clip_out
            )

        # -------------------------------------------------
        # Original classifier
        # -------------------------------------------------

        fused = torch.cat(
            [
                clip_out,
                freq_out,
            ],
            dim=1,
        )

        logits = self.model.classifier(
            fused
        )

        return logits


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    model,
    test_loader,
    fake_label,
    real_label,
):
    """
    Evaluate using the project's existing convention:

        FAKE = 0
        REAL = 1

    sigmoid(logit) = P(REAL)

    threshold 0.5 therefore produces:

        >= 0.5 -> REAL (1)
        <  0.5 -> FAKE (0)
    """

    all_probs = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE
            )

            with torch.amp.autocast(
                "cuda",
                enabled=(
                    DEVICE.type == "cuda"
                ),
            ):

                logits = model(images)

            # IMPORTANT:
            # This is P(REAL), because label 1 = REAL.
            probs_real = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
                .flatten()
            )

            all_probs.extend(
                probs_real.tolist()
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

    labels = np.array(
        all_labels
    )

    probs_real = np.array(
        all_probs
    )

    # -----------------------------------------------------
    # Convert P(REAL) into class prediction
    # -----------------------------------------------------

    preds = (
        probs_real >= THRESHOLD
    ).astype(int)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        labels,
        preds,
    )

    precision = precision_score(
        labels,
        preds,
        pos_label=fake_label,
        zero_division=0,
    )

    recall = recall_score(
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
        labels=[
            fake_label,
            real_label,
        ],
    )

    # With matrix order:
    #
    #              Pred FAKE  Pred REAL
    # Actual FAKE     cm[0,0]   cm[0,1]
    # Actual REAL     cm[1,0]   cm[1,1]
    #
    # FN = actual FAKE predicted REAL
    # FP = actual REAL predicted FAKE

    fn = int(cm[0, 1])
    fp = int(cm[1, 0])

    # -----------------------------------------------------
    # Also calculate P(FAKE)
    # -----------------------------------------------------

    probs_fake = 1.0 - probs_real

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fn": fn,
        "fp": fp,
        "cm": cm,
        "mean_p_fake": probs_fake[
            labels == fake_label
        ].mean(),
        "mean_p_fake_real_images": probs_fake[
            labels == real_label
        ].mean(),
    }


# =========================================================
# LOAD GENERATOR DATASET
# =========================================================

def load_generator(
    generator,
):
    """
    Load the same GenImage test split used by
    eval_cross_generator.py.
    """

    config_path = os.path.join(
        PROJECT_ROOT,
        "config.yaml",
    )

    with open(
        config_path,
        "r",
    ) as f:

        config = yaml.safe_load(f)

    config["data"]["batch_size"] = (
        BATCH_SIZE
    )

    config["data"]["num_workers"] = (
        NUM_WORKERS
    )

    config["data"]["genimage_path"] = (
        f"D:/IAD_data/genimage_{generator}"
    )

    print()
    print(
        f"Loading generator: "
        f"{generator.upper()}"
    )

    print(
        f"Dataset path: "
        f"{config['data']['genimage_path']}"
    )

    (
        _,
        _,
        test_loader,
        class_to_idx,
    ) = get_genimage_dataloaders(
        config
    )

    fake_label = class_to_idx[
        "FAKE"
    ]

    real_label = class_to_idx[
        "REAL"
    ]

    print(
        f"class_to_idx: "
        f"{class_to_idx}"
    )

    return (
        test_loader,
        fake_label,
        real_label,
    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_result(
    generator,
    mode,
    result,
):

    print()
    print(
        "-" * 90
    )

    print(
        f"{generator.upper()} | "
        f"{mode}"
    )

    print(
        "-" * 90
    )

    print(
        f"Accuracy:              "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Precision (FAKE+):     "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall (FAKE+):        "
        f"{result['recall']:.4f}"
    )

    print(
        f"F1 (FAKE+):            "
        f"{result['f1']:.4f}"
    )

    print(
        f"FN (missed FAKE):      "
        f"{result['fn']}"
    )

    print(
        f"FP (false accusation): "
        f"{result['fp']}"
    )

    print(
        f"Mean P(FAKE), actual FAKE: "
        f"{result['mean_p_fake']:.4f}"
    )

    print(
        f"Mean P(FAKE), actual REAL: "
        f"{result['mean_p_fake_real_images']:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        "                 "
        "Pred FAKE   Pred REAL"
    )

    cm = result["cm"]

    print(
        f"Actual FAKE     "
        f"{cm[0,0]:>6}      "
        f"{cm[0,1]:>6}"
    )

    print(
        f"Actual REAL     "
        f"{cm[1,0]:>6}      "
        f"{cm[1,1]:>6}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 90
    )

    print(
        "IAD-V2 BRANCH ABLATION"
    )

    print(
        "=" * 90
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Primary threshold: {THRESHOLD}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "FAKE = 0"
    )

    print(
        "REAL = 1"
    )

    print(
        "sigmoid(logit) = P(REAL)"
    )

    print(
        "=" * 90
    )


    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    print()
    print(
        "Loading existing checkpoint..."
    )

    base_model = FusionDetector(
        "openai/clip-vit-base-patch16"
    ).to(DEVICE)

    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    base_model.load_state_dict(
        state
    )

    base_model.eval()

    print(
        "Checkpoint loaded."
    )


    # =====================================================
    # GENERATORS
    # =====================================================

    generators = [
        "sd15",
        "biggan",
        "adm",
    ]

    modes = [
        "full",
        "clip_only",
        "frequency_only",
    ]


    # =====================================================
    # RESULTS
    # =====================================================

    all_results = {}


    # =====================================================
    # RUN ABLATIONS
    # =====================================================

    for generator in generators:

        (
            test_loader,
            fake_label,
            real_label,
        ) = load_generator(
            generator
        )

        all_results[
            generator
        ] = {}

        for mode in modes:

            print()
            print(
                "=" * 90
            )

            print(
                f"Evaluating "
                f"{generator.upper()} "
                f"| {mode}"
            )

            print(
                "=" * 90
            )

            ablated_model = (
                AblationDetector(
                    base_model,
                    mode=mode,
                ).to(DEVICE)
            )

            result = evaluate(
                ablated_model,
                test_loader,
                fake_label,
                real_label,
            )

            all_results[
                generator
            ][mode] = result

            print_result(
                generator,
                mode,
                result,
            )


    # =====================================================
    # SUMMARY TABLE
    # =====================================================

    print()
    print()
    print(
        "=" * 110
    )

    print(
        "BRANCH ABLATION SUMMARY"
    )

    print(
        "=" * 110
    )

    header = (
        f"{'Generator':<12}"
        f"{'Mode':<18}"
        f"{'Prec':>8}"
        f"{'Recall':>8}"
        f"{'F1':>8}"
        f"{'Acc':>8}"
        f"{'FN':>8}"
        f"{'FP':>8}"
    )

    print(header)

    print(
        "-" * 110
    )

    for generator in generators:

        for mode in modes:

            result = all_results[
                generator
            ][mode]

            print(
                f"{generator.upper():<12}"
                f"{mode:<18}"
                f"{result['precision']:>8.4f}"
                f"{result['recall']:>8.4f}"
                f"{result['f1']:>8.4f}"
                f"{result['accuracy']:>8.4f}"
                f"{result['fn']:>8}"
                f"{result['fp']:>8}"
            )

    print(
        "=" * 110
    )


    # =====================================================
    # SAVE TEXT REPORT
    # =====================================================

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    report_path = os.path.join(
        output_dir,
        "branch_ablation_results.txt",
    )

    with open(
        report_path,
        "w",
    ) as f:

        f.write(
            "IAD-V2 Branch Ablation Results\n"
        )

        f.write(
            "========================================\n\n"
        )

        f.write(
            "Checkpoint: "
            f"{CHECKPOINT}\n"
        )

        f.write(
            "Threshold: 0.5\n"
        )

        f.write(
            "FAKE label: 0\n"
        )

        f.write(
            "REAL label: 1\n"
        )

        f.write(
            "sigmoid(logit) = P(REAL)\n\n"
        )

        for generator in generators:

            f.write(
                f"\n{generator.upper()}\n"
            )

            f.write(
                "-" * 50
                + "\n"
            )

            for mode in modes:

                result = all_results[
                    generator
                ][mode]

                f.write(
                    f"\n{mode}\n"
                )

                f.write(
                    f"Accuracy: {result['accuracy']:.4f}\n"
                )

                f.write(
                    f"Precision: {result['precision']:.4f}\n"
                )

                f.write(
                    f"Recall: {result['recall']:.4f}\n"
                )

                f.write(
                    f"F1: {result['f1']:.4f}\n"
                )

                f.write(
                    f"FN: {result['fn']}\n"
                )

                f.write(
                    f"FP: {result['fp']}\n"
                )

    print()
    print(
        f"Saved report:\n"
        f"{report_path}"
    )

    print()
    print(
        "ABLATION COMPLETE."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()