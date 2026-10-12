import os, sys, torch, yaml
from PIL import Image
from torchvision import transforms
sys.path.insert(0, '.')
import transformers.utils.import_utils, transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from src.models.detector import FusionDetector
from transformers import CLIPImageProcessor

with open('config.yaml') as f:
    config = yaml.safe_load(f)
device = torch.device('cuda')
model = FusionDetector(config['model']['clip_model_name']).to(device)
model.load_state_dict(torch.load('outputs/genimage_multigen3_fusion.pth', map_location=device, weights_only=True))
model.eval()

processor = CLIPImageProcessor.from_pretrained(config['model']['clip_model_name'])
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=processor.image_mean, std=processor.image_std)])

fake_dir = 'D:/IAD_data/genimage_biggan/FAKE'
real_dir = 'D:/IAD_data/genimage_biggan/REAL'
fakes = sorted(os.listdir(fake_dir))[:5]
reals = sorted(os.listdir(real_dir))[:5]

correct = 0
total = 0
for label, d, files in [('FAKE', fake_dir, fakes), ('REAL', real_dir, reals)]:
    for fn in files:
        img = Image.open(os.path.join(d, fn)).convert('RGB')
        t = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logit = model(t)
            p_real = torch.sigmoid(logit).item()
        verdict = 'AI-GENERATED' if p_real < 0.5 else 'REAL'
        ok = (label == 'FAKE' and verdict == 'AI-GENERATED') or (label == 'REAL' and verdict == 'REAL')
        correct += int(ok)
        total += 1
        status = "OK" if ok else "MISS"
        print(f'[{label}] {fn} -> {verdict}  P(Real)={p_real:.4f}  {status}')
print(f'\n{correct}/{total} correct')
