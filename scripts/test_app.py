import sys; sys.path.append('.')
import torch, yaml, cv2
from PIL import Image
from torchvision import transforms
from app import load_model, build_inference_transform, ViTGradCAM

def run():
    model, device, clip_mean, clip_std, _ = load_model()
    tf = build_inference_transform(clip_mean, clip_std)
    img = Image.open('D:/IAD_data/genimage_biggan/FAKE/biggan_fake_00000.png').convert('RGB')
    t = tf(img).unsqueeze(0).to(device)
    cam_gen = ViTGradCAM(model)
    cam, log = cam_gen.generate(t)
    print(f'Test success! logit {log}, cam shape {cam.shape}')

if __name__ == '__main__':
    run()
