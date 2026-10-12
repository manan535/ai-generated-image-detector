import os
import sys
import yaml
import torch
import datetime
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from transformers import CLIPImageProcessor

CHECKPOINT = "outputs/genimage_multigen4_vqdm_fusion.pth"

def verify_checkpoint():
    abs_path = os.path.abspath(CHECKPOINT)
    print(f"Absolute path: {abs_path}")
    print(f"Exists: {os.path.exists(abs_path)}")
    
    if not os.path.exists(abs_path):
        return

    mtime = os.path.getmtime(abs_path)
    mtime_dt = datetime.datetime.fromtimestamp(mtime)
    print(f"Modification time: {mtime_dt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    
    print("\n--- Checkpoint Metadata ---")
    keys = list(checkpoint.keys())
    print(f"Top-level keys: {keys}")
    
    print(f"training_generators: {checkpoint.get('training_generators')}")
    print(f"held_out_generators: {checkpoint.get('held_out_generators')}")
    print(f"epochs: {checkpoint.get('epochs')}")
    print(f"batch_size: {checkpoint.get('batch_size')}")
    print(f"learning_rate: {checkpoint.get('learning_rate')}")
        
    print("\n--- Model Architecture & Sanity Test ---")
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        
    clip_model_name = config["model"]["clip_model_name"]
    model = FusionDetector(clip_model_name).to(device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded from 'model_state_dict'")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded from raw checkpoint dict")
        
    model.eval()
    
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    
    img_path = 'D:/IAD_data/genimage_sd15/FAKE/' + os.listdir('D:/IAD_data/genimage_sd15/FAKE')[0]
    raw_img = Image.open(img_path).convert("RGB")
    tensor = transform(raw_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(tensor)
    
    logit = logits[0, 0].item()
    prob_real = torch.sigmoid(torch.tensor(logit)).item()
    prob_fake = 1.0 - prob_real
    threshold = 0.5
    is_fake = prob_real < threshold
    prediction = "AI-GENERATED (FAKE)" if is_fake else "REAL"
    
    print(f"\nTest Image: {img_path}")
    print(f"Logit: {logit:.6f}")
    print(f"P(REAL): {prob_real:.6f}")
    print(f"P(FAKE): {prob_fake:.6f}")
    print(f"Threshold: {threshold}")
    print(f"Final Prediction: {prediction}")

if __name__ == "__main__":
    verify_checkpoint()
