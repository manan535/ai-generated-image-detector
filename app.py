"""
app.py — AI Image Authenticity Detector

Final inference demo using:
    FusionDetector
    CLIP ViT-B/16 + 2D-FFT frequency branch

Final checkpoint:
    outputs/genimage_multigen4_vqdm_fusion.pth

Training generators:
    SD1.5 + BigGAN + VQDM

Held-out generators:
    ADM + Wukong
"""

import io
import yaml
import torch
import numpy as np
import cv2
import streamlit as st

from PIL import Image
from torchvision import transforms

import transformers.utils.import_utils
import transformers.modeling_utils

# Transformers / Torch compatibility workaround
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.models.detector import FusionDetector

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT = "outputs/genimage_multigen4_vqdm_fusion.pth"
THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ViT Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        # Final CLIP vision transformer layer
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

        # Positive logit -> REAL, Negative logit -> FAKE
        if logit.item() >= 0:
            logit.backward()
        else:
            (-logit).backward()

        acts = self.activations
        grads = self.gradients

        if acts.dim() == 2:
            acts = acts.unsqueeze(0)
        if isinstance(grads, tuple):
            grads = grads[0]
        if grads.dim() == 2:
            grads = grads.unsqueeze(0)

        # Remove CLS token
        grads = grads[0, 1:, :]
        acts = acts[0, 1:, :]

        weights = torch.mean(grads, dim=0)
        cam = torch.matmul(acts, weights)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        cam = cam.view(14, 14).detach().cpu().numpy()

        return cam, logit.item()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    clip_model_name = config["model"]["clip_model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionDetector(clip_model_name).to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    clip_mean = processor.image_mean
    clip_std = processor.image_std

    return model, device, clip_mean, clip_std, clip_model_name


import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_inference_transform(clip_mean, clip_std):
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=clip_mean, std=clip_std),
        ToTensorV2(),
    ])

import base64
def get_image_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()



# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page
# ─────────────────────────────────────────────────────────────────────────────


st.set_page_config(
    page_title="AI Image Authenticity Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS to eliminate scrolling
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)





# Load model
model, device, clip_mean, clip_std, clip_model_name = load_model()
transform = build_inference_transform(clip_mean, clip_std)


# ─────────────────────────────────────────────────────────────────────────────
# Image uploader
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛡️ AI Authenticity")
    st.markdown("Upload an image to detect if it's AI-generated or real.")
    uploaded = st.file_uploader(
        "Select Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )
    st.divider()
    with st.expander("Model Details"):
        st.markdown("**Architecture:** FusionDetector")
        st.markdown("**Backbone:** CLIP ViT-B/16 + 2D-FFT")
        st.markdown("**Threshold:** 0.5")

if uploaded is not None:
    # Load image
    raw_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    orig_w, orig_h = raw_img.size

    # Preprocess & Inference
    image_np = np.array(raw_img)
    tensor = transform(image=image_np)["image"]
    input_tensor = tensor.unsqueeze(0).to(device)

    cam_generator = ViTGradCAM(model)
    cam_mask, logit_val = cam_generator.generate(input_tensor)

    # Probabilities
    prob_real = torch.sigmoid(torch.tensor(logit_val)).item()
    prob_fake = 1.0 - prob_real
    is_fake = prob_real < THRESHOLD

    verdict = "AI-GENERATED" if is_fake else "REAL"
    confidence = prob_fake if is_fake else prob_real

    # Grad-CAM overlay
    cam_mask_resized = cv2.resize(cam_mask, (orig_w, orig_h))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask_resized), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    
    orig_img_np = np.array(raw_img)
    overlay_img = cv2.addWeighted(orig_img_np, 0.5, cam_heatmap, 0.5, 0)
    overlay_img = Image.fromarray(overlay_img)

    # ─────────────────────────────────────────────────────────────────────────
    # UI Layout: Side-by-side
    # ─────────────────────────────────────────────────────────────────────────
    col_img, col_res = st.columns([1.5, 1], gap="large")

    with col_img:
        st.subheader("Analyzed Image")
        img_b64 = get_image_base64(raw_img)
        stamp_class = "stamp-fake" if is_fake else "stamp-real"
        stamp_text = "FAKE" if is_fake else "REAL"
        
        html_code = f"""
        <style>
        @keyframes stampAnimation {{
            0% {{ opacity: 0; transform: translate(-50%, -50%) scale(3) rotate(-15deg); }}
            50% {{ opacity: 1; transform: translate(-50%, -50%) scale(0.9) rotate(-15deg); }}
            100% {{ opacity: 0.85; transform: translate(-50%, -50%) scale(1) rotate(-15deg); }}
        }}
        @keyframes stampAnimationReal {{
            0% {{ opacity: 0; transform: translate(-50%, -50%) scale(3) rotate(10deg); }}
            50% {{ opacity: 1; transform: translate(-50%, -50%) scale(0.9) rotate(10deg); }}
            100% {{ opacity: 0.85; transform: translate(-50%, -50%) scale(1) rotate(10deg); }}
        }}
        .image-container {{
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: rgba(0,0,0,0.02);
            border-radius: 8px;
            padding: 10px;
            height: 80vh;
            overflow: hidden;
            border: 1px solid rgba(0,0,0,0.1);
        }}
        @media (prefers-color-scheme: dark) {{
            .image-container {{
                background-color: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.1);
            }}
        }}
        .uploaded-image {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 4px;
        }}
        .stamp {{
            position: absolute;
            top: 50%;
            left: 50%;
            font-size: clamp(3rem, 6vw, 6rem);
            font-weight: 900;
            text-transform: uppercase;
            border: clamp(5px, 1vw, 10px) solid;
            border-radius: 15px;
            padding: 10px 40px;
            pointer-events: none;
            box-shadow: 0 0 20px rgba(0,0,0,0.1), inset 0 0 20px rgba(0,0,0,0.1);
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            z-index: 10;
        }}
        .stamp-fake {{
            color: #ff3b30;
            border-color: #ff3b30;
            background-color: rgba(255, 59, 48, 0.1);
            animation: stampAnimation 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) 0.3s forwards;
            opacity: 0;
        }}
        .stamp-real {{
            color: #34c759;
            border-color: #34c759;
            background-color: rgba(52, 199, 89, 0.1);
            animation: stampAnimationReal 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) 0.3s forwards;
            opacity: 0;
        }}
        </style>
        
        <div class="image-container">
            <img src="data:image/jpeg;base64,{img_b64}" class="uploaded-image">
            <div class="stamp {stamp_class}">{stamp_text}</div>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

    with col_res:
        st.subheader("Analysis Result")
        
        if is_fake:
            st.error(f"### {verdict}\n**Confidence: {confidence:.1%}**", icon="🤖")
        else:
            st.success(f"### {verdict}\n**Confidence: {confidence:.1%}**", icon="📸")
            
        st.progress(prob_fake, text=f"P(AI-Generated) = {prob_fake:.1%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**Model Attention / Grad-CAM**")
        img_cam_b64 = get_image_base64(overlay_img)
        st.markdown(
            f'<div style="height: 25vh; display: flex; justify-content: left;">'
            f'<img src="data:image/jpeg;base64,{img_cam_b64}" style="max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 8px;">'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption("ℹ️ *Grad-CAM highlights regions influencing the model's prediction.*")
        
    # ─────────────────────────────────────────────────────────────────────────
    # Technical Details (Expandable)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ View Technical Details & Model Context"):
        st.markdown(f"**Training:** SD1.5 + BigGAN + VQDM | **Held-out evaluation:** ADM + Wukong")
        
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            st.markdown(f"""
            - **Architecture:** FusionDetector  
            - **CLIP backbone:** `{clip_model_name}` (Frozen)
            - **Frequency branch:** 2D-FFT → CNN → 128-dim  
            - **Fusion dim:** 896 (768 + 128)  
            """)
        with det_col2:
            st.markdown(f"""
            - **Checkpoint:** `{CHECKPOINT}`  
            - **Decision threshold:** {THRESHOLD}  
            - **Input size:** 224 × 224 px  
            - **Loss:** BCEWithLogitsLoss  
            """)

else:
    st.info("Upload an image to begin.", icon="📂")

st.divider()
st.caption("AI Image Authenticity Detector")