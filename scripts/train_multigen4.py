import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# Torch/Transformers compatibility workaround
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from src.data.multigen4_dataset import get_multigen4_dataloaders


def main():

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config["data"]["batch_size"] = 16
    config["data"]["num_workers"] = 0

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    print("\n======================================")
    print("GenImage Multi-Generator Experiment 4")
    print("======================================")

    print("\nTraining generators:")
    print("  - SD1.5")
    print("  - BigGAN")
    print("  - VQDM")

    print("\nHeld-out generators:")
    print("  - ADM")
    print("  - Wukong")

    print("\nArchitecture:")
    print("  - CLIP ViT-B/16")
    print("  - Frequency branch")
    print("  - 896-dim fusion")
    print("  - MLP classifier")

    print("\n======================================\n")

    # Load data
    train_loader, val_loader, test_datasets = (
        get_multigen4_dataloaders(config)
    )

    print(
        f"Train size: {len(train_loader.dataset)} images"
    )

    print(
        f"Validation size: {len(val_loader.dataset)} images"
    )

    # Initialize model
    model = FusionDetector(
        config["model"]["clip_model_name"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    num_epochs = 10

    print(
        "\n=== Experiment 4 Training: 10 Epochs ==="
    )

    best_val_f1 = -1.0
    best_epoch = -1

    os.makedirs("outputs", exist_ok=True)

    save_path = (
        "outputs/genimage_multigen4_vqdm_fusion.pth"
    )

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            ).float().unsqueeze(1)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda")
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = (
            running_loss / len(train_loader)
        )

        epoch_time = time.time() - start_time

        print(
            f"\n--- Epoch {epoch + 1} Complete ---"
        )

        print(
            f"Time: {epoch_time:.2f}s"
        )

        print(
            f"Average Loss: {epoch_loss:.4f}"
        )

        if device.type == "cuda":

            peak_vram = (
                torch.cuda.max_memory_allocated(device)
                / (1024 * 1024)
            )

            print(
                f"Peak VRAM: {peak_vram:.2f} MB"
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )

    # Save final checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_generators": [
                "sd15",
                "biggan",
                "vqdm",
            ],
            "held_out_generators": [
                "adm",
                "wukong",
            ],
            "epochs": num_epochs,
            "batch_size": 16,
            "learning_rate": 1e-4,
        },
        save_path
    )

    print("\n======================================")
    print("EXPERIMENT 4 TRAINING COMPLETE")
    print("======================================")

    print(
        f"Checkpoint saved to:\n{save_path}"
    )


if __name__ == "__main__":
    main()