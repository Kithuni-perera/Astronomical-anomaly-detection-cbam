<div align="center">

# Real-Time Unsupervised Astronomical Anomaly Detection
### CBAM-Enhanced Convolutional Autoencoders & Latent-Space Isolation Forests on SDSS DR17

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/SDSS-DR17%20SkyServer-4B0082?style=for-the-badge)](https://www.sdss.org/dr17/)
[![Paper](https://img.shields.io/badge/IEEE%20Manuscript-PDF-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](IEEE_Astronomical_Anomaly_Detection.pdf)

*A lightweight, single-pass deep learning pipeline for real-time celestial outlier discovery and automated sky-survey quality control.*

</div>

---

## Executive Summary
Modern wide-field astronomical surveys such as the **Sloan Digital Sky Survey (SDSS DR17)** and upcoming **Vera C. Rubin Observatory (LSST)** generate petabytes of imaging data containing billions of celestial sources. Identifying rare astrophysical phenomena—such as **high-redshift quasars ($z > 3.0$)**, **ultra-faint low-surface-brightness (LSB) galaxies**, and **unflagged optical ghosting artifacts**—requires fast, unsupervised computer vision pipelines that operate without scarce spectroscopic labels.

While generative adversarial baselines (e.g., WGAN-GP) achieve strong morphological outlier detection, they suffer from unstable multi-hour training and slow iterative latent-space optimization (`~1,250 ms/image`). This project introduces a **CBAM-Enhanced Convolutional Autoencoder** paired with a **Latent-Space Isolation Forest**, matching WGAN-GP accuracy (**0.8592 ROC-AUC**, **89.94% Precision**) while running **$>30\times$ faster** (`39.36 ms/image`).

---

## System Architecture

```text
SDSS DR17 Cutout (3x64x64) 
       │
       ▼  [Per-Image Min-Max Normalization]
┌──────────────────────────────────────────────────────────┐
│  Convolutional Encoder (Conv2d + BatchNorm + LeakyReLU)  │
│  64x64x3 ──► 32x32x32 ──► 16x16x64 ──► 8x8x128           │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  CBAM Attention Bottleneck (8x8x128)                     │
│  ├── Channel Attention (128 -> 32 -> 128 Sigmoid Gate)   │
│  └── Spatial Attention (7x7 Conv Sky-Noise Suppression)  │
└─────────────┬──────────────────────────────┬─────────────┘
              │                              │
              ▼ [Global AvgPool]             ▼ [Transposed Conv Decoder]
    128-D Morphological Latent       Reconstructed Cutout (3x64x64)
              │                              │
              ▼                              ▼
    Isolation Forest (200 Trees)    Top-10% Peak Structural Residual
              │                              │
              └──────────────┬───────────────┘
                             ▼
               Hybrid Anomaly Score & Ranking
