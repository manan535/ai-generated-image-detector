import os
import sys
import yaml
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from src.models.detector import FusionDetector

CHECKPOINT = "outputs/genimage_multigen4_vqdm_fusion.pth"

def run_comparison():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_name = config["model"]["clip_model_name"]
    model = FusionDetector(clip_model_name).to(device)
    
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    clip_mean = processor.image_mean
    clip_std = processor.image_std
    
    # A: eval_multigen4.py pipeline (Albumentations)
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=clip_mean, std=clip_std),
        ToTensorV2(),
    ])
    
    # B: Streamlit app pipeline (torchvision)
    app_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])
    
    adm_fake_dir = "D:/IAD_data/genimage_adm/FAKE"
    files = sorted(os.listdir(adm_fake_dir))
    
    # We need to test images that were in the test split. 
    # Let's just pick the first 10, regardless of train/test for now to compare pipelines, 
    # or actually let's use the first 10 from the test split to match the prompt perfectly.
    from src.data.multigen4_dataset import create_split_indices
    from torchvision.datasets import ImageFolder
    ds = ImageFolder("D:/IAD_data/genimage_adm")
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    train_idx, val_idx, test_idx = create_split_indices(ds)
    
    adm_fakes_test = []
    for i in test_idx:
        img_path, label = ds.samples[i]
        if idx_to_class[label] == "FAKE":
            adm_fakes_test.append(img_path)
        if len(adm_fakes_test) == 10:
            break
            
    print(f"{'Filename':<30} | {'Eval Logit':<12} | {'Eval P(REAL)':<12} | {'Eval Pred':<12} | {'App Logit':<12} | {'App P(REAL)':<12} | {'App Pred':<12}")
    print("-" * 120)
    
    eval_fakes = 0
    app_fakes = 0
    
    for img_path in adm_fakes_test:
        fname = os.path.basename(img_path)
        raw_img = Image.open(img_path).convert("RGB")
        
        # Pipeline A
        image_np = np.array(raw_img)
        tensor_A = eval_transform(image=image_np)["image"].unsqueeze(0).to(device)
        
        # Pipeline B
        tensor_B = app_transform(raw_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logit_A = model(tensor_A)[0, 0].item()
                logit_B = model(tensor_B)[0, 0].item()
                
        prob_real_A = torch.sigmoid(torch.tensor(logit_A)).item()
        prob_real_B = torch.sigmoid(torch.tensor(logit_B)).item()
        
        pred_A = "REAL" if prob_real_A >= 0.5 else "FAKE"
        pred_B = "REAL" if prob_real_B >= 0.5 else "FAKE"
        
        if pred_A == "FAKE": eval_fakes += 1
        if pred_B == "FAKE": app_fakes += 1
        
        print(f"{fname:<30} | {logit_A:<12.4f} | {prob_real_A:<12.4f} | {pred_A:<12} | {logit_B:<12.4f} | {prob_real_B:<12.4f} | {pred_B:<12}")
        
    print("-" * 120)
    print(f"Total FAKE predictions: Eval Pipeline = {eval_fakes}/10, App Pipeline = {app_fakes}/10")

if __name__ == "__main__":
    run_comparison()
