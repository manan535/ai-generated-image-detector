import os
import sys
import io
import torch
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from src.data.multigen_dataset import create_split_dataset
from src.models.detector import FusionDetector


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CLIP_MODEL = "openai/clip-vit-base-patch16"

BATCH_SIZE = 16

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "genimage_multigen_fusion.pth"
)

SD15_PATH = "D:/IAD_data/genimage_sd15"
BIGGAN_PATH = "D:/IAD_data/genimage_biggan"
ADM_PATH = "D:/IAD_data/genimage_adm"

JPEG_QUALITIES = [75, 50, 30]


# --------------------------------------------------
# JPEG dataset wrapper
# --------------------------------------------------

class JPEGDataset(Dataset):

    def __init__(self, base_dataset, quality):
        self.base_dataset = base_dataset
        self.quality = quality

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):

        image, label = self.base_dataset[idx]

        # The base dataset has already applied CLIP normalization,
        # so we first convert the tensor back approximately to
        # an image representation for JPEG simulation.
        image = image.clone()

        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            dtype=image.dtype
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            dtype=image.dtype
        ).view(3, 1, 1)

        image = image * std + mean
        image = image.clamp(0, 1)

        image = (
            image.permute(1, 2, 0)
            .mul(255)
            .byte()
            .numpy()
        )

        pil_image = Image.fromarray(image)

        buffer = io.BytesIO()

        pil_image.save(
            buffer,
            format="JPEG",
            quality=self.quality
        )

        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")

        compressed = np.array(compressed).astype(
            np.float32
        ) / 255.0

        # Re-apply CLIP normalization
        compressed = (
            compressed -
            np.array([
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
        ).permute(2, 0, 1).float()

        return compressed, label


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

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

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            # P(REAL) >= 0.5 -> REAL
            predictions.extend(
                (probabilities >= 0.5)
                .long()
                .cpu()
                .numpy()
            )

            labels.extend(
                targets.numpy()
            )

    labels = np.array(labels)
    predictions = np.array(predictions)

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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    print("\nLoading checkpoint:")
    print(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model = FusionDetector(
        CLIP_MODEL
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print("Checkpoint loaded successfully.")

    # create_split_dataset only needs the model
    # configuration for CLIP normalization.
    config = {
        "model": {
            "clip_model_name": CLIP_MODEL
        }
    }

    generators = {
        "SD1.5": SD15_PATH,
        "BigGAN": BIGGAN_PATH,
        "ADM": ADM_PATH
    }

    results = {}

    # --------------------------------------------------
    # Evaluate each generator
    # --------------------------------------------------

    for generator_name, generator_path in generators.items():

        print("\n===================================")
        print(generator_name)
        print("===================================")

        _, _, test_dataset = create_split_dataset(
            generator_path,
            config
        )

        # Original
        print("\nOriginal")

        original_metrics = evaluate(
            model,
            test_dataset,
            device
        )

        results[
            (generator_name, "Original")
        ] = original_metrics

        print(
            f"Accuracy: {original_metrics['accuracy']:.4f}"
        )

        # JPEG qualities
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
                f"Accuracy: {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f}"
            )

            print(
                f"Recall: {metrics['recall']:.4f}"
            )

            print(
                f"F1: {metrics['f1']:.4f}"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n\n===================================")
    print("JPEG ROBUSTNESS SUMMARY")
    print("===================================")

    print(
        "\nGenerator   Original   Q75      Q50      Q30"
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

        print(
            f"{generator_name:<11}"
            f"{original:.4f}     "
            f"{q75:.4f}   "
            f"{q50:.4f}   "
            f"{q30:.4f}"
        )

        drop = (
            original - q50
        ) * 100

        print(
            f"  Q50 accuracy drop: "
            f"{drop:.2f} percentage points"
        )

    print("\nAcceptance criterion:")
    print(
        "JPEG Q50 accuracy drop <= 15 percentage points."
    )


if __name__ == "__main__":
    main()