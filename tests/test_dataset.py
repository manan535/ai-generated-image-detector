import os
import tempfile
import yaml
import pytest
from unittest.mock import patch
import torch
from PIL import Image

from src.data.dataset import get_dataloaders

@pytest.fixture
def dummy_cifake_dir():
    # Create a temporary directory structure mimicking CIFAKE
    temp_dir = tempfile.TemporaryDirectory()
    base_path = temp_dir.name
    
    # Create train and test splits with FAKE and REAL subdirectories
    for split in ["train", "test"]:
        for label in ["FAKE", "REAL"]:
            dir_path = os.path.join(base_path, split, label)
            os.makedirs(dir_path)
            
            # Create 10 dummy images per class per split (total 40 images)
            for i in range(10):
                img = Image.new('RGB', (32, 32), color=(73, 109, 137))
                img.save(os.path.join(dir_path, f"img_{i}.jpg"))
                
    yield base_path
    temp_dir.cleanup()

@pytest.fixture
def mock_config(dummy_cifake_dir):
    return {
        "data": {
            "cifake_path": dummy_cifake_dir,
            "batch_size": 4,
            "num_workers": 0
        },
        "model": {
            "clip_model_name": "openai/clip-vit-base-patch16"
        }
    }

@patch("src.data.dataset.download_cifake")
def test_dataset_loading_and_splitting(mock_download, mock_config, caplog):
    import logging
    caplog.set_level(logging.INFO)
    # Call the dataloader function
    train_loader, val_loader, test_loader = get_dataloaders(mock_config)
    
    # 1. Verify download was called (or checked)
    mock_download.assert_called_once_with(mock_config["data"]["cifake_path"])
    
    # 2. Verify dataset sizes (Total 40 images -> 70% train (28), 15% val (6), 15% test (6))
    assert len(train_loader.dataset) == 28
    assert len(val_loader.dataset) == 6
    assert len(test_loader.dataset) == 6
    
    # 3. Verify batch loading and B2 preprocessing (224x224, Normalized)
    train_batch, train_labels = next(iter(train_loader))
    assert train_batch.shape == (4, 3, 224, 224)
    assert train_labels.shape == (4,)
    
    # 4. Print class balance check (already logged in get_dataloaders, we can check logs)
    print("\n--- Per-Split Class Balance ---")
    for record in caplog.records:
        if "split:" in record.message:
            print(record.message)
            
    # Explicitly verify the log contains the stratification
    assert any("Train split" in rec.message for rec in caplog.records)
    assert any("Val split" in rec.message for rec in caplog.records)
    assert any("Test split" in rec.message for rec in caplog.records)
