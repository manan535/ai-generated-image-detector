import os
import sys
import yaml
import torch
import numpy as np
import cv2
import csv
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from src.data.multigen_dataset import create_split_indices
from transformers import CLIPImageProcessor

CHECKPOINT = "outputs/genimage_multigen4_vqdm_fusion.pth"
OUT_DIR = "outputs/gradcam_validation"
THRESHOLD = 0.5

# Reusing ViTGradCAM from app.py
class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.clip.encoder.layers[-1].layer_norm1
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        logit = logits[0, 0]

        if logit.item() >= 0:
            logit.backward()
        else:
            (-logit).backward()

        acts = self.activations
        grads = self.gradients
        if acts.dim() == 2: acts = acts.unsqueeze(0)
        if isinstance(grads, tuple): grads = grads[0]
        if grads.dim() == 2: grads = grads.unsqueeze(0)
        grads = grads[0, 1:, :]
        acts = acts[0, 1:, :]
        weights = torch.mean(grads, dim=0)
        cam = torch.matmul(acts, weights)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        cam = cam.view(14, 14).detach().cpu().numpy()
        return cam, logit.item()

def run_validation():
    os.makedirs(OUT_DIR, exist_ok=True)
    
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
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])
    
    cam_generator = ViTGradCAM(model)
    
    generators = {
        'SD15': config['data']['sd15_path'],
        'ADM': config['data']['adm_path'],
        'Wukong': config['data']['wukong_path']
    }
    
    csv_file = open(os.path.join(OUT_DIR, "validation_summary.csv"), "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["generator", "filename", "ground_truth", "prediction", "confidence", "correct", "image_path", "gradcam_path"])
    
    results = []
    
    for gen_name, path in generators.items():
        dataset = ImageFolder(path)
        # FAKE=0, REAL=1
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        
        train_idx, val_idx, test_idx = create_split_indices(dataset)
        
        # We need 5 correct predictions (try to balance FAKE/REAL if possible)
        selected = []
        counts = {'FAKE': 0, 'REAL': 0}
        
        for i in test_idx:
            img_path, label = dataset.samples[i]
            gt_label = idx_to_class[label]
            
            # Skip if we already have enough of this class (aim for 2-3 of each)
            if counts[gt_label] >= 3 and sum(counts.values()) < 5:
                continue
                
            raw_img = Image.open(img_path).convert("RGB")
            tensor = transform(raw_img)
            input_tensor = tensor.unsqueeze(0).to(device)
            
            cam_mask, logit_val = cam_generator.generate(input_tensor)
            prob_real = torch.sigmoid(torch.tensor(logit_val)).item()
            is_fake = prob_real < THRESHOLD
            pred_label = "FAKE" if is_fake else "REAL"
            
            if pred_label == gt_label:
                orig_w, orig_h = raw_img.size
                confidence = (1.0 - prob_real) if is_fake else prob_real
                
                # Save Original
                base_name = f"{gen_name}_{len(selected)+1:02d}"
                orig_save_path = os.path.join(OUT_DIR, f"{base_name}_original.png")
                raw_img.save(orig_save_path)
                
                # Save Grad-CAM
                cam_mask_resized = cv2.resize(cam_mask, (orig_w, orig_h))
                cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask_resized), cv2.COLORMAP_JET)
                cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(np.array(raw_img), 0.5, cam_heatmap, 0.5, 0)
                overlay_img = Image.fromarray(overlay)
                cam_save_path = os.path.join(OUT_DIR, f"{base_name}_gradcam.png")
                overlay_img.save(cam_save_path)
                
                csv_writer.writerow([gen_name, os.path.basename(img_path), gt_label, pred_label, f"{confidence:.4f}", "True", orig_save_path, cam_save_path])
                
                selected.append({
                    'gen': gen_name,
                    'orig': raw_img,
                    'cam': overlay_img,
                    'gt': gt_label,
                    'pred': pred_label,
                    'conf': confidence
                })
                counts[gt_label] += 1
                
                if len(selected) == 5:
                    break
                    
        results.extend(selected)
        print(f"Collected {len(selected)} for {gen_name}. FAKE: {counts['FAKE']}, REAL: {counts['REAL']}")

    csv_file.close()
    
    # Create contact sheet
    fig, axes = plt.subplots(5, 6, figsize=(20, 18))
    fig.suptitle('Grad-CAM Validation - Correctly Classified Examples', fontsize=20)
    
    for i, res in enumerate(results):
        row = i % 5
        col_offset = (i // 5) * 2
        
        ax_orig = axes[row, col_offset]
        ax_cam = axes[row, col_offset + 1]
        
        ax_orig.imshow(res['orig'])
        ax_orig.axis('off')
        if row == 0:
            ax_orig.set_title(f"{res['gen']} - Original", fontsize=14)
            
        ax_cam.imshow(res['cam'])
        ax_cam.axis('off')
        ax_cam.set_title(f"GT: {res['gt']} | Pred: {res['pred']}\nConf: {res['conf']:.2%}", fontsize=12)
        if row == 0:
            ax_cam.set_title(f"{res['gen']} - Grad-CAM\nGT: {res['gt']} | Pred: {res['pred']}\nConf: {res['conf']:.2%}", fontsize=12)
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    contact_path = os.path.join(OUT_DIR, "contact_sheet.png")
    plt.savefig(contact_path, bbox_inches='tight', dpi=150)
    print(f"Saved contact sheet to {contact_path}")

if __name__ == '__main__':
    run_validation()
