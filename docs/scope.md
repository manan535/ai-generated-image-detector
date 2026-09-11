# Project Scope & Feasibility

## Objective
To build a highly robust, explainable classification system capable of distinguishing between real photographs and AI-generated images. The project prioritizes generalization to unseen generators and robustness against common digital degradations (e.g., resizing, JPEG compression).

## Technical Approach
**Dual-Branch Architecture:**
1. **CLIP ViT-B/16 Vision Backbone:** Leverages the robust, pre-trained representations of OpenAI's CLIP to detect semantic inconsistencies and spatial artifacts. Output size: 768 dimensions.
2. **Frequency Analysis Branch (2D-FFT):** Calculates the 2D Fast Fourier Transform and log-spectrum of the image to explicitly capture periodic upsampling artifacts missed by standard spatial CNNs/ViTs. Output size: 128 dimensions.
3. **Fusion Head:** Concatenates both branches into an 896-dimensional vector, processed by a fully connected classification head with threshold tuning via ROC curves.

## Datasets
1. **CIFAKE (Local Dev):** Used strictly for pipeline smoke-testing. It provides a lightweight way to verify data loaders, augmentation, and training loops locally without GPU constraints.
2. **GenImage (Production):** The primary training and evaluation dataset. Contains 2.68M images from 8 different generators (including Midjourney and Stable Diffusion). We will extract a small stratified subset (e.g., 5K real / 5K fake from Stable Diffusion) for debugging, and use the full dataset for final training on Colab/Kaggle.

## Success Criteria & Validation
To ensure the model is practically useful and "honest" about its capabilities, the following criteria must be met:
1. **In-Distribution Baseline:** ≥85% accuracy and F1 score measured on the held-out test split of the primary training generator (e.g., Stable Diffusion).
2. **Cross-Generator Evaluation (OOD):** ≥75% accuracy when evaluated on images from unseen generators (e.g., tested on Midjourney when trained on Stable Diffusion). This is the primary measure of real-world generalization.
3. **Compression Robustness:** No more than a 15-point drop from in-distribution accuracy when evaluated on images subjected to JPEG quality 50.
4. **Explainability (Grad-CAM):** The system must include a Grad-CAM overlay to visually explain *why* an image was flagged. The heatmap must center on a plausible artifact region in at least 12 of 15 correct predictions during qualitative review.

## Out of Scope & Feasibility Risks
**Explicitly Out of Scope:**
- Video deepfake detection
- Audio deepfake detection
- Real-time or low-latency infrastructure
- Adversarial robustness (e.g., defending against explicit evasion attacks)

**Primary Feasibility Risk:**
The main risk is that strong in-distribution accuracy can easily mask poor real-world generalization, which is exactly why the cross-generator and compression robustness validation steps are mandatory for success.
