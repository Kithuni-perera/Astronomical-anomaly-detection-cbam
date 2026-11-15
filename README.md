\# Real-Time Unsupervised Astronomical Anomaly Detection Using CBAM-Enhanced Convolutional Autoencoders and Latent Isolation Forests



\[!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

\[!\[Dataset](https://img.shields.io/badge/Dataset-SDSS%20DR17-blueviolet.svg)](https://www.sdss.org/dr17/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)



\## Overview

Modern wide-field astronomical sky surveys such as the \*\*Sloan Digital Sky Survey (SDSS DR17)\*\* and upcoming \*\*Vera C. Rubin Observatory (LSST)\*\* generate petabytes of imaging data containing billions of celestial objects\[cite: 1]. Discovering rare astrophysical phenomena—such as \*\*high-redshift quasars ($z > 3.0$)\*\*, \*\*low-surface-brightness (LSB) galaxies\*\*, and \*\*unflagged optical ghosting artifacts\*\*—requires fast, unsupervised deep learning methods that do not rely on scarce spectroscopic labels\[cite: 1].



This repository contains the official implementation and research manuscript for a lightweight, real-time unsupervised anomaly detection pipeline combining a \*\*Convolutional Block Attention Module (CBAM)-Enhanced Convolutional Autoencoder\*\* with a \*\*Latent-Space Isolation Forest\*\*\[cite: 1].



\---



\## Key Results \& Benchmarking

Evaluated across \*\*647 multi-band SDSS DR17 cutouts ($64 \\times 64$ pixels)\*\*, our proposed hybrid architecture achieves WGAN-level detection accuracy while operating \*\*$>30\\times$ faster\*\* during inference\[cite: 1]:



| Model Architecture | ROC-AUC | Precision | Recall | F1-Score | Train Time (s) | Inference (ms/img) |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| WGAN-GP Baseline (Margalef-Bentabol et al., 2020) | 0.8600 | 0.8000 | 0.8000 | 0.8000 | 14,400.00 | 1,250.00 |

| Baseline Standard CAE (No Attention) | 0.7983 | 0.8513 | 0.8300 | 0.8405 | \*\*36.09\*\* | \*\*38.30\*\* |

| \*\*Proposed Attention-Enhanced CAE (CBAM)\*\* | \*\*0.8592\*\* | \*\*0.8994\*\* | 0.8050 | \*\*0.8496\*\* | 38.07 | 39.36 |



\### Highlights

\* \*\*+6.09% ROC-AUC Gain:\*\* Embedding CBAM channel and spatial attention at the encoder bottleneck suppresses empty background sky noise and isolates core galactic morphology\[cite: 1].

\* \*\*Bias-Mitigated Latent Scoring:\*\* Combining per-image min-max dynamic range normalization, top-$10\\%$ structural peak residuals, and a $128$-D Latent Isolation Forest eliminates brightness/angular-size biases that cause standard autoencoders to miss compact quasars\[cite: 1].

\* \*\*Dual-Purpose Blind Sky Discovery:\*\* Scanning \*\*250 spectroscopically unclassified SDSS DR17 sources\*\* successfully isolated high-redshift quasars ($z > 3.01$), ultra-faint LSB galaxy candidates, and unflagged optical ghosting artifacts\[cite: 1].



\---



\## Visual Results



\### 1. Ablation Study: ROC Curve Comparison

!\[ROC Curve Comparison](04\_Results\_and\_Paper/roc\_curve\_comparison.png)



\### 2. Blind Discovery on Unclassified SDSS DR17 Sources

!\[Novel Discoveries Gallery](04\_Results\_and\_Paper/novel\_discoveries\_gallery.png)



\---



\## Repository Structure



```text

Astronomical-anomaly-detection-cbam/

├── IEEE\_Astronomical\_Anomaly\_Detection.pdf   # Full IEEE Conference Research Manuscript

├── README.md                                 # Project documentation \& benchmarks

├── 03\_Models\_and\_Code/

│   ├── step3\_build\_dataset.py                # Automated SDSS DR17 SkyServer SQL query \& cutout downloader

│   ├── step4\_train\_attention.py              # CBAM-Enhanced Convolutional Autoencoder training pipeline

│   ├── step5\_detect\_anomalies.py             # Real-time inference \& spectroscopic outlier ranking

│   ├── step6\_evaluate\_metrics.py             # Controlled ablation study \& WGAN-GP benchmarking

│   └── step7\_discover\_unknowns.py            # Blind sky scanning on unclassified SDSS DR17 sources

└── 04\_Results\_and\_Paper/

&#x20;   ├── roc\_curve\_comparison.png              # Ablation ROC curves

&#x20;   ├── novel\_discoveries\_gallery.png         # Top 6 blind sky discoveries

&#x20;   ├── top\_anomalies\_visualization.png       # Input vs. reconstruction vs. residual heatmaps

&#x20;   ├── training\_loss\_curve.png               # Model convergence curve

&#x20;   └── model\_comparison\_table.csv            # Quantitative benchmark metrics

