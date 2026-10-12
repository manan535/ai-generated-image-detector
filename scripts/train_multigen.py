import os
import sys
import yaml
import torch

from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from src.data.multigen_dataset import (
    get_multigen_dataloaders
)

from src.models.detector import FusionDetector


CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "config.yaml"
)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            probs = torch.sigmoid(logits).squeeze(1)

            preds = (
                probs >= 0.5
            ).long()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training"
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.float().to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda")
        ):

            logits = model(images).squeeze(1)

            loss = criterion(
                logits,
                labels
            )

        if device.type == "cuda":

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=loss.item()
        )

    return running_loss / len(loader)


def main():

    config = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    print("\n===================================")
    print("MULTI-GENERATOR TRAINING")
    print("===================================")

    print("Training generators:")
    print("  - SD1.5")
    print("  - BigGAN")

    print("\nHeld-out generator:")
    print("  - ADM")

    print("\nArchitecture:")
    print("  - CLIP ViT-B/16")
    print("  - Frequency branch")
    print("  - 896-dim fusion")
    print("  - MLP classifier")

    print("\n===================================\n")

    (
        train_loader,
        val_loader,
        _,
        _,
    ) = get_multigen_dataloaders(
        config
    )

    print(
        f"Training batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = FusionDetector(
        config["model"]["clip_model_name"]
    )

    model = model.to(device)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = AdamW(
        trainable_params,
        lr=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    epochs = 10

    best_f1 = -1.0

    os.makedirs(
        os.path.join(
            PROJECT_ROOT,
            "outputs"
        ),
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "genimage_multigen_fusion.pth"
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(epochs):

        print(
            f"\n========== Epoch "
            f"{epoch + 1}/{epochs} =========="
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        metrics = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"\nTrain Loss: {train_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Val Precision (FAKE): "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Val Recall (FAKE): "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Val F1 (FAKE): "
            f"{metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch + 1,

                    "val_metrics":
                        metrics,

                    "training_generators":
                        [
                            "sd15",
                            "biggan"
                        ],

                    "held_out_generator":
                        "adm",
                },
                checkpoint_path
            )

            print(
                "\nBest checkpoint saved:"
            )

            print(
                checkpoint_path
            )

    print("\n===================================")
    print("TRAINING COMPLETE")
    print("===================================")

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )


if __name__ == "__main__":
    main()