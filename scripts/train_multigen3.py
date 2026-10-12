import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import transformers.utils.import_utils, transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from src.data.multigen3_dataset import get_multigen3_dataloaders


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    config['data']['batch_size'] = 16
    config['data']['num_workers'] = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_datasets = get_multigen3_dataloaders(config)

    print(f"Train size: {len(train_loader.dataset)} images")
    print(f"Val size: {len(val_loader.dataset)} images")

    # Initialize model
    model = FusionDetector(config['model']['clip_model_name']).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = 10
    print("\n=== GenImage Multi-Gen Experiment 3: Training (10 Epochs) ===")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
        print(f"--- Epoch {epoch+1} Complete ---")
        print(f"Time: {epoch_time:.2f}s  |  Avg Loss: {epoch_loss:.4f}  |  Peak VRAM: {peak_vram:.2f} MB")

    os.makedirs('outputs', exist_ok=True)
    save_path = 'outputs/genimage_multigen3_fusion.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nCheckpoint saved to {save_path}")

if __name__ == '__main__':
    main()