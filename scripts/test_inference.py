"""Quick headless test: load checkpoint, run inference on one image, print result."""
import os, sys, torch, yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import transformers.utils.import_utils, transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from transformers import CLIPImageProcessor

with open('config.yaml') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionDetector(config['model']['clip_model_name']).to(device)
model.load_state_dict(torch.load('outputs/genimage_multigen3_fusion.pth', map_location=device, weights_only=True))
model.eval()
print(f"Checkpoint loaded on {device}")

processor = CLIPImageProcessor.from_pretrained(config['model']['clip_model_name'])
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

# Pick one FAKE and one REAL image from the SD1.5 test set
fake_dir = 'D:/IAD_data/genimage_sd15/FAKE'
real_dir = 'D:/IAD_data/genimage_sd15/REAL'

fake_img_path = os.path.join(fake_dir, os.listdir(fake_dir)[0])
real_img_path = os.path.join(real_dir, os.listdir(real_dir)[0])

for label, path in [("FAKE", fake_img_path), ("REAL", real_img_path)]:
    img = Image.open(path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logit = model(tensor)
        p_real = torch.sigmoid(logit).item()
    p_fake = 1.0 - p_real
    verdict = "AI-GENERATED" if p_real < 0.5 else "REAL"
    print(f"[{label}] {os.path.basename(path)}  ->  Verdict: {verdict}  P(Real)={p_real:.4f}  P(Fake)={p_fake:.4f}")
