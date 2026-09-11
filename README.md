# AI-Generated Image Detection

This project aims to detect AI-generated images using a robust dual-branch architecture. By combining spatial, semantic, and frequency-domain features, the model is designed to catch multiple types of generation artifacts, making it resilient to both distribution shifts and image degradations like JPEG compression.

## Architecture
The system employs a dual-branch neural network:
1. **Image Branch:** A frozen CLIP ViT-B/16 vision encoder fine-tuned to capture high-level semantic and pixel-level artifacts. This branch outputs a 768-dimensional feature vector.
2. **Frequency Branch:** A 2D Fast Fourier Transform (FFT) combined with log-spectrum analysis, processed by a lightweight CNN/MLP. This is explicitly designed to catch periodic upsampling artifacts typical of generative models. This branch outputs a 128-dimensional vector.

The features are concatenated into an 896-dimensional fused vector and passed through a classification head to output the final fake/real probability.

## Datasets
- **Primary Training (Remote):** [GenImage](https://github.com/GenImage-Dataset/GenImage), containing over 2.6 million images across 8 generators (including Stable Diffusion and Midjourney). This dataset will be used for the actual model training on cloud compute (e.g., Colab/Kaggle).
- **Smoke-Testing (Local):** [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), a tiny 120K image dataset used exclusively for local mechanical pipeline testing and debugging to avoid large downloads.

## Setup & Testing
To set up the environment locally for CPU smoke-testing:
```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Unix: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -r requirements.txt
```
Run the smoke test to verify the model architecture:
```bash
python -m unittest tests/smoke_test.py
```

See `docs/scope.md` for detailed project objectives and success criteria.
