"""
Step 3: General-Purpose AI Comparison — Image Selection & Template Creation

Selects 10 images from the Experiment 4 test split, runs our model,
and creates a comparison template for manual ChatGPT/Claude evaluation.

Distribution:
  3 REAL SD1.5, 2 FAKE SD1.5
  1 REAL ADM,   2 FAKE ADM
  1 REAL Wukong, 1 FAKE Wukong
"""

import os
import sys
import csv
import yaml
import shutil
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from src.data.multigen_dataset import create_split_indices
from transformers import CLIPImageProcessor

CHECKPOINT = "outputs/genimage_multigen4_vqdm_fusion.pth"
OUT_DIR = "outputs/general_ai_comparison"
IMG_DIR = os.path.join(OUT_DIR, "comparison_images")

# Target distribution: (generator_key, config_key, class_name, count)
TARGETS = [
    ("SD15",   "sd15_path",   "REAL", 3),
    ("SD15",   "sd15_path",   "FAKE", 2),
    ("ADM",    "adm_path",    "REAL", 1),
    ("ADM",    "adm_path",    "FAKE", 2),
    ("Wukong", "wukong_path", "REAL", 1),
    ("Wukong", "wukong_path", "FAKE", 1),
]


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_name = config["model"]["clip_model_name"]

    # Load model
    model = FusionDetector(clip_model_name).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Transform (Albumentations, matching eval pipeline)
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=processor.image_mean, std=processor.image_std),
        ToTensorV2(),
    ])

    rows = []
    image_counter = 0

    for gen_name, config_key, target_class, count in TARGETS:
        data_path = config["data"][config_key]
        dataset = ImageFolder(data_path)
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        _, _, test_idx = create_split_indices(dataset)

        collected = 0
        for i in test_idx:
            if collected >= count:
                break

            img_path, label = dataset.samples[i]
            gt_class = idx_to_class[label]

            if gt_class != target_class:
                continue

            # Run inference
            raw_img = Image.open(img_path).convert("RGB")
            image_np = np.array(raw_img)
            tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logit = model(tensor)[0, 0].item()

            prob_real = torch.sigmoid(torch.tensor(logit)).item()
            prob_fake = 1.0 - prob_real
            is_fake = prob_real < 0.5
            our_pred = "AI-GENERATED" if is_fake else "REAL"
            our_conf = prob_fake if is_fake else prob_real

            # Copy image with a clean sequential name
            image_counter += 1
            orig_ext = os.path.splitext(img_path)[1]
            clean_name = f"img_{image_counter:02d}_{gen_name}_{target_class}{orig_ext}"
            dest_path = os.path.join(IMG_DIR, clean_name)
            shutil.copy2(img_path, dest_path)

            rows.append({
                "filename": clean_name,
                "original_filename": os.path.basename(img_path),
                "generator": gen_name,
                "ground_truth": target_class,
                "our_prediction": our_pred,
                "our_confidence": f"{our_conf:.1%}",
                "chatgpt_prediction": "",
                "chatgpt_confidence": "",
                "claude_prediction": "",
                "claude_confidence": "",
                "notes": "",
            })

            collected += 1

        print(f"{gen_name} {target_class}: collected {collected}/{count}")

    # Write CSV
    csv_path = os.path.join(OUT_DIR, "comparison_template.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "original_filename", "generator", "ground_truth",
            "our_prediction", "our_confidence",
            "chatgpt_prediction", "chatgpt_confidence",
            "claude_prediction", "claude_confidence",
            "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to: {csv_path}")

    # Write README
    readme_path = os.path.join(OUT_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""\
GENERAL-PURPOSE AI COMPARISON — Instructions
=============================================

Purpose:
  Test whether general-purpose AI systems (ChatGPT, Claude) can reliably
  classify images as AI-generated vs real, compared with our purpose-built
  FusionDetector.

Folder contents:
  comparison_images/      — 10 test images
  comparison_template.csv — Results template (our model columns filled in)
  README.txt              — This file

Standardized prompt (use EXACTLY this for every image, for both systems):
----------------------------------------------------------------------

  Is this image AI-generated or a real-world/camera-captured image?

  Give:
  1. Your classification: AI-generated or Real
  2. Your confidence from 0-100%

  Do not use external image-search or provenance information.
  Judge only the image itself.

----------------------------------------------------------------------

Procedure:
  1. Upload each image from comparison_images/ one at a time.
  2. Paste the standardized prompt above.
  3. Record the system's classification and confidence in the CSV.
  4. Do NOT provide any hints about the ground truth or our model's answer.
  5. Use the same prompt for every image. Do not modify it.
  6. If the system refuses to answer or hedges, record what it says in "notes".

Image distribution:
  3 REAL SD1.5    (in-distribution for our model)
  2 FAKE SD1.5    (in-distribution for our model)
  1 REAL ADM      (held-out generator for our model)
  2 FAKE ADM      (held-out generator for our model)
  1 REAL Wukong   (held-out generator for our model)
  1 FAKE Wukong   (held-out generator for our model)

Ground truth labels are known and recorded in the CSV.
Images were selected from the Experiment 4 test split, not cherry-picked
based on our model's predictions.
""")

    print(f"README written to: {readme_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'#':<4} {'Filename':<40} {'Gen':<8} {'GT':<6} {'Our Pred':<15} {'Our Conf':<10}")
    print(f"{'='*90}")
    for i, r in enumerate(rows, 1):
        print(f"{i:<4} {r['filename']:<40} {r['generator']:<8} {r['ground_truth']:<6} {r['our_prediction']:<15} {r['our_confidence']:<10}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
