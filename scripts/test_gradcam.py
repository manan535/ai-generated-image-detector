import os, sys, torch, yaml, cv2
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import transformers.utils.import_utils, transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector
from transformers import CLIPImageProcessor

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        # Target the last layer_norm1 before MLP or layer_norm2 
        # layer_norm1 takes [1, 197, 768]
        self.target_layer = model.clip.encoder.layers[-1].layer_norm1
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # output shape: [1, 197, 768]
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] shape: [1, 197, 768]
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        
        # We must enable requires_grad on input tensor to propagate gradients backwards
        input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        logit = logits[0, 0]
        
        # We want to explain whatever class the model is leaning towards.
        # BCEWithLogitsLoss: lower logit (<0) = FAKE (label 0), higher logit (>0) = REAL (label 1).
        # We take gradient of logit if logit > 0 (REAL), or gradient of -logit if logit < 0 (FAKE)
        if logit.item() > 0:
            logit.backward()
        else:
            (-logit).backward()
        acts = self.activations
        grads = self.gradients
        
        print("acts dim:", acts.dim(), "shape:", acts.shape)
        if type(grads) == tuple:
            print("grads is tuple:", len(grads))
            grads = grads[0]
        print("grads dim:", grads.dim(), "shape:", grads.shape)
        
        # Ensure batch dim exists
        if acts.dim() == 2: acts = acts.unsqueeze(0)
        if grads.dim() == 2: grads = grads.unsqueeze(0)
        
        # Ignore CLS token (index 0) -> take spatial tokens (1 to 196)
        grads = grads[0, 1:, :] # [196, 768]
        acts = acts[0, 1:, :] # [196, 768]
        
        # Global average pooling of gradients
        weights = torch.mean(grads, dim=0) # [768]
        
        # Weighted combination of activations
        cam = torch.matmul(acts, weights) # [196]
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        # Reshape to 14x14
        cam = cam.view(14, 14).detach().cpu().numpy()
        return cam, logit.item()


with open('config.yaml') as f:
    config = yaml.safe_load(f)
device = torch.device('cuda')
model = FusionDetector(config['model']['clip_model_name']).to(device)
model.load_state_dict(torch.load('outputs/genimage_multigen3_fusion.pth', map_location=device, weights_only=True))

processor = CLIPImageProcessor.from_pretrained(config['model']['clip_model_name'])
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=processor.image_mean, std=processor.image_std)])

fake_img_path = 'D:/IAD_data/genimage_biggan/FAKE/' + os.listdir('D:/IAD_data/genimage_biggan/FAKE')[0]
img = Image.open(fake_img_path).convert('RGB')
t = tf(img).unsqueeze(0).to(device)

cam_gen = ViTGradCAM(model)
cam, logit = cam_gen.generate(t)

print(f"Logit: {logit}")
print(f"CAM shape: {cam.shape}, min: {cam.min()}, max: {cam.max()}")
