"""
Experiment 4 — JPEG Robustness Evaluation

Final checkpoint:
    outputs/genimage_multigen4_vqdm_fusion.pth

Generators:
    SD1.5
    BigGAN
    VQDM
    ADM
    Wukong

JPEG conditions:
    Original
    Q75
    Q50
    Q30

Threshold:
    0.5

Acceptance criterion:
    Q50 accuracy drop <= 15 percentage points
"""

import os
import sys
import io

import yaml
import torch
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ─────────────────────────────────────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ─────────────────────────────────────────────────────────────────────────────
# Transformers / Torch compatibility workaround
# ─────────────────────────────────────────────────────────────────────────────

import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = (
    lambda: None
)

transformers.modeling_utils.check_torch_load_is_safe = (
    lambda: None
)


# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────

from src.data.multigen4_dataset import (
    create_split_dataset
)

from src.models.detector import FusionDetector


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CLIP_MODEL = "openai/clip-vit-base-patch16"

BATCH_SIZE = 16

THRESHOLD = 0.5

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "genimage_multigen4_vqdm_fusion.pth"
)

CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "config.yaml"
)

JPEG_QUALITIES = [
    75,
    50,
    30
]


# ─────────────────────────────────────────────────────────────────────────────
# JPEG dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class JPEGDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        quality
    ):

        self.base_dataset = base_dataset
        self.quality = quality

    def __len__(self):

        return len(
            self.base_dataset
        )

    def __getitem__(self, idx):

        image, label = self.base_dataset[idx]

        # The base dataset has already applied
        # CLIP normalization.
        #
        # Convert back approximately to RGB,
        # JPEG-compress, then re-apply CLIP normalization.

        image = image.clone()

        mean = torch.tensor(
            [
                0.48145466,
                0.4578275,
                0.40821073
            ],
            dtype=image.dtype
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [
                0.26862954,
                0.26130258,
                0.27577711
            ],
            dtype=image.dtype
        ).view(
            3,
            1,
            1
        )

        # Undo CLIP normalization
        image = image * std + mean

        image = image.clamp(
            0,
            1
        )

        # CHW -> HWC
        image = (
            image
            .permute(1, 2, 0)
            .mul(255)
            .byte()
            .numpy()
        )

        pil_image = Image.fromarray(
            image
        )

        # JPEG encode
        buffer = io.BytesIO()

        pil_image.save(
            buffer,
            format="JPEG",
            quality=self.quality
        )

        buffer.seek(0)

        # JPEG decode
        compressed = Image.open(
            buffer
        ).convert("RGB")

        compressed = np.array(
            compressed
        ).astype(
            np.float32
        ) / 255.0

        # Re-apply CLIP normalization
        compressed = (
            compressed
            - np.array([
                0.48145466,
                0.4578275,
                0.40821073
            ])
        ) / np.array([
            0.26862954,
            0.26130258,
            0.27577711
        ])

        compressed = torch.from_numpy(
            compressed
        ).permute(
            2,
            0,
            1
        ).float()

        return compressed, label


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    dataset,
    device
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    labels = []
    predictions = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(
                images
            )

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            # P(REAL) >= 0.5 -> REAL (1)
            # P(REAL) <  0.5 -> FAKE (0)

            predictions.extend(
                (
                    probabilities >= THRESHOLD
                )
                .long()
                .cpu()
                .numpy()
            )

            labels.extend(
                targets.numpy()
            )

    labels = np.array(
        labels
    )

    predictions = np.array(
        predictions
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():

    print("\n==============================================")
    print("EXPERIMENT 4 — JPEG ROBUSTNESS")
    print("==============================================")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Checkpoint:\n{CHECKPOINT_PATH}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Load config
    # ─────────────────────────────────────────────────────────────────────────

    with open(
        CONFIG_PATH,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    config["data"]["batch_size"] = BATCH_SIZE
    config["data"]["num_workers"] = 0


    # ─────────────────────────────────────────────────────────────────────────
    # Load checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    print(
        "\nLoading checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = FusionDetector(
        config["model"]["clip_model_name"]
    )

    # Experiment 4 checkpoint contains metadata
    # plus model_state_dict.
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )


    # ─────────────────────────────────────────────────────────────────────────
    # Generators
    # ─────────────────────────────────────────────────────────────────────────

    generators = {
        "SD1.5": config["data"]["sd15_path"],
        "BigGAN": config["data"]["biggan_path"],
        "VQDM": config["data"]["vqdm_path"],
        "ADM": config["data"]["adm_path"],
        "Wukong": config["data"]["wukong_path"],
    }


    results = {}


    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate each generator
    # ─────────────────────────────────────────────────────────────────────────

    for generator_name, generator_path in generators.items():

        print(
            "\n=============================================="
        )

        print(
            f"{generator_name}"
        )

        print(
            "=============================================="
        )

        print(
            f"Path: {generator_path}"
        )

        # Create exactly the same deterministic
        # 70/15/15 split used by Experiment 4.
        _, _, test_dataset = create_split_dataset(
            generator_path,
            config
        )

        print(
            f"Test images: {len(test_dataset)}"
        )


        # ─────────────────────────────────────────────────────────────────────
        # Original
        # ─────────────────────────────────────────────────────────────────────

        print(
            "\nOriginal"
        )

        original_metrics = evaluate(
            model,
            test_dataset,
            device
        )

        results[
            (generator_name, "Original")
        ] = original_metrics

        print(
            f"Accuracy:  {original_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {original_metrics['precision']:.4f}"
        )

        print(
            f"Recall:    {original_metrics['recall']:.4f}"
        )

        print(
            f"F1:        {original_metrics['f1']:.4f}"
        )


        # ─────────────────────────────────────────────────────────────────────
        # JPEG conditions
        # ─────────────────────────────────────────────────────────────────────

        for quality in JPEG_QUALITIES:

            print(
                f"\nJPEG Quality {quality}"
            )

            jpeg_dataset = JPEGDataset(
                test_dataset,
                quality
            )

            metrics = evaluate(
                model,
                jpeg_dataset,
                device
            )

            results[
                (generator_name, f"Q{quality}")
            ] = metrics

            print(
                f"Accuracy:  {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f}"
            )

            print(
                f"Recall:    {metrics['recall']:.4f}"
            )

            print(
                f"F1:        {metrics['f1']:.4f}"
            )


    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    print(
        "\n\n=============================================="
    )

    print(
        "JPEG ROBUSTNESS SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        "\nGenerator    Original    Q75       Q50       Q30"
    )

    print(
        "-" * 58
    )

    for generator_name in generators:

        original = results[
            (generator_name, "Original")
        ]["accuracy"]

        q75 = results[
            (generator_name, "Q75")
        ]["accuracy"]

        q50 = results[
            (generator_name, "Q50")
        ]["accuracy"]

        q30 = results[
            (generator_name, "Q30")
        ]["accuracy"]

        drop_q50 = (
            original - q50
        ) * 100

        drop_q75 = (
            original - q75
        ) * 100

        drop_q30 = (
            original - q30
        ) * 100

        print(
            f"{generator_name:<11}"
            f"{original:.4f}     "
            f"{q75:.4f}     "
            f"{q50:.4f}     "
            f"{q30:.4f}"
        )

        print(
            f"  Q75 drop: {drop_q75:+.2f} pp"
        )

        print(
            f"  Q50 drop: {drop_q50:+.2f} pp"
        )

        print(
            f"  Q30 drop: {drop_q30:+.2f} pp"
        )


    # ─────────────────────────────────────────────────────────────────────────
    # Acceptance criterion
    # ─────────────────────────────────────────────────────────────────────────

    print(
        "\n=============================================="
    )

    print(
        "ACCEPTANCE CRITERION"
    )

    print(
        "=============================================="
    )

    print(
        "JPEG Q50 accuracy drop <= 15 percentage points."
    )

    print(
        "\nThis criterion is evaluated separately for"
        " each generator."
    )

    for generator_name in generators:

        original = results[
            (generator_name, "Original")
        ]["accuracy"]

        q50 = results[
            (generator_name, "Q50")
        ]["accuracy"]

        drop = (
            original - q50
        ) * 100

        status = (
            "PASS"
            if drop <= 15
            else "FAIL"
        )

        print(
            f"{generator_name:<11} "
            f"{drop:>7.2f} pp  [{status}]"
        )

    print(
        "\n=============================================="
    )


if __name__ == "__main__":
    main()