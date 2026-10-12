# AI-Generated Image Detection

This project aims to detect AI-generated images using a dual-branch architecture that combines visual features with frequency-domain features.

The main focus is not only in-distribution accuracy, but **generalization to image generators that were never seen during training**. The evaluation also measures robustness to common image degradation such as JPEG compression.

## Architecture

The system uses a dual-branch neural network:

1. **Visual Branch**

   A frozen **CLIP ViT-B/16** vision encoder extracts high-level visual features from the input image.

   - Input: `224 × 224 × 3`
   - Output: **768-dimensional feature vector**
   - CLIP parameters remain frozen during detector training.

2. **Frequency Branch**

   The input image is converted to grayscale and transformed into the frequency domain using a **2D FFT**. The magnitude spectrum is log-transformed and processed by a lightweight CNN.

   - Input: frequency-domain representation
   - Output: **128-dimensional feature vector**
   - Designed to investigate whether frequency-domain information provides complementary signals for AI-generated image detection.

3. **Feature Fusion**

   The two feature vectors are concatenated:

   `768 + 128 = 896 dimensions`

   The fused representation is passed through a classification head consisting of:

   - Linear layer: `896 → 256`
   - ReLU
   - Dropout
   - Linear layer: `256 → 1`
   - Sigmoid for the final probability.

The final output is a binary **REAL / FAKE** prediction together with a confidence score.

## Datasets

### GenImage

The primary dataset source is
[GenImage](https://github.com/GenImage-Dataset/GenImage), a large-scale benchmark containing real and AI-generated images from multiple generation methods.

For controlled experiments, the project uses selected GenImage-derived subsets rather than the entire dataset.

Current experiments include:

- **Stable Diffusion 1.5 (SD1.5)**
- **BigGAN**
- **VQDM**
- **ADM**
- **Wukong**

The main generalization experiment trains on:

**Training:**
- SD1.5
- BigGAN
- VQDM

**Held out during training:**
- ADM
- Wukong

This allows the model to be evaluated on generators it has never seen during training.

### CIFAKE

[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) is used only for **local pipeline smoke-testing and debugging**.

It is not used as the primary training benchmark because it is based on a much smaller image resolution and a more limited generation setup.

## Evaluation

The project evaluates both in-distribution performance and cross-generator generalization.

The latest multi-generator experiment produced:

| Test Generator | Training Exposure | Accuracy |
|---|---|---:|
| SD1.5 | Seen | **94.17%** |
| BigGAN | Seen | **93.67%** |
| VQDM | Seen | **93.17%** |
| ADM | **Unseen** | **72.00%** |
| Wukong | **Unseen** | **85.33%** |

The most important finding is the improvement in unseen ADM performance as training diversity increased:

| Training Configuration | Unseen ADM Accuracy |
|---|---:|
| SD1.5 only | **49.0%** |
| SD1.5 + BigGAN | **60.5%** |
| SD1.5 + BigGAN + VQDM | **72.0%** |

This provides evidence that **increasing generator diversity during training improves cross-generator generalization**.

The current project target for cross-generator accuracy is **75%**, so the 72% ADM result represents substantial improvement while still leaving a measurable gap.

## Robustness

The model is also evaluated under JPEG compression to measure the effect of common real-world image degradation.

The evaluation includes JPEG quality levels such as:

- Original
- Quality 75
- Quality 50
- Quality 30

The goal is to determine whether detector performance remains stable when images are compressed or degraded.

## Setup & Testing

To set up the environment locally for CPU smoke-testing:

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Unix
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -r requirements.txt