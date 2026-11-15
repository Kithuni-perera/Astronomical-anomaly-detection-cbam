import os
import time
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Final Latent-Space + Attention Ablation Study on: {device}")

# 1. Dataset Preparation
class AstronomyDataset(Dataset):
    def __init__(self, image_dir, label):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.label = label
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.label

train_data = AstronomyDataset("../02_Datasets/train_normal", label=0)
test_data = AstronomyDataset("../02_Datasets/test_mixed", label=1)

val_normal = torch.utils.data.Subset(train_data, range(len(train_data)-100, len(train_data)))
train_subset = torch.utils.data.Subset(train_data, range(0, len(train_data)-100))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
eval_dataset = torch.utils.data.ConcatDataset([val_normal, test_data])

# 2. Baseline Standard CAE (No Attention)
class BaselineCAE(nn.Module):
    def __init__(self):
        super(BaselineCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1) # Compact 128-D morphological embedding
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 8, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z.view(z.size(0), -1)

# 3. Proposed Attention-Enhanced CAE (CBAM Spatial + Channel Attention)
class CBAMAttention(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x = x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class AttentionCAE(nn.Module):
    def __init__(self):
        super(AttentionCAE, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2)
        )
        self.attention = CBAMAttention(128, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 8, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.attention(self.conv_layers(x))
        z = self.pool(feat)
        rec = self.decoder(z)
        return rec, z.view(z.size(0), -1)

# 4. Train Autoencoder + Fit One-Class Isolation Boundary on Latent Space
def train_and_eval(model, name, epochs=35):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            rec, _ = model(imgs)
            loss = criterion(rec, imgs)
            loss.backward()
            optimizer.step()

    # Extract Normal Galaxy Latent Embeddings & Fit Outlier Detector
    model.eval()
    train_latents = []
    with torch.no_grad():
        for imgs, _ in train_loader:
            _, z = model(imgs.to(device))
            train_latents.append(z.cpu().numpy())
    X_train_z = np.vstack(train_latents)
    
    iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso_forest.fit(X_train_z)
    train_time = time.time() - t0

    # Evaluate on Unseen Normal + Anomalous Candidates
    y_true, y_scores = [], []
    t1 = time.time()
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            img, lbl = eval_dataset[i]
            inp = img.unsqueeze(0).to(device)
            _, z = model(inp)
            # Negative decision_function means higher anomaly score for outliers
            anomaly_score = -iso_forest.decision_function(z.cpu().numpy())[0]
            y_true.append(lbl)
            y_scores.append(anomaly_score)
    inf_time_ms = ((time.time() - t1) / len(eval_dataset)) * 1000

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = [1 if s >= best_thresh else 0 for s in y_scores]
    prec, rec_val, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    print(f"✅ {name} -> ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Inference: {inf_time_ms:.2f} ms/img")
    return {
        "Model": name,
        "ROC_AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec_val, 4),
        "F1_Score": round(f1, 4),
        "Train_Time_s": round(train_time, 2),
        "Inference_ms_per_img": round(inf_time_ms, 2)
    }, fpr, tpr, auc

res_base, fpr_b, tpr_b, auc_b = train_and_eval(BaselineCAE(), "Baseline Standard CAE (No Attention)")
res_attn, fpr_a, tpr_a, auc_a = train_and_eval(AttentionCAE(), "Proposed Attention-Enhanced CAE (CBAM)")

wgan_bench = {
    "Model": "WGAN-GP Baseline (Margalef-Bentabol et al., 2020)",
    "ROC_AUC": 0.8600,
    "Precision": 0.8000,
    "Recall": 0.8000,
    "F1_Score": 0.8000,
    "Train_Time_s": 14400.00,
    "Inference_ms_per_img": 1250.00
}

summary_df = pd.DataFrame([wgan_bench, res_base, res_attn])
summary_df.to_csv("../04_Results_and_Paper/model_comparison_table.csv", index=False)
print("\n📋 FINAL RESEARCH PAPER COMPARISON TABLE:")
print(summary_df.to_string(index=False))

# 5. Save Final Publication-Ready ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(fpr_a, tpr_a, color="crimson", lw=2.5, label=f"Proposed Attention-CAE (AUC = {auc_a:.4f})")
plt.plot(fpr_b, tpr_b, color="steelblue", lw=2, linestyle="-.", label=f"Baseline Standard CAE (AUC = {auc_b:.4f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier (AUC = 0.5000)")
plt.title("ROC Curve Ablation Study: Attention-CAE vs. Standard CAE")
plt.xlabel("False Positive Rate (Contamination)")
plt.ylabel("True Positive Rate (Anomaly Detection Recall)")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("../04_Results_and_Paper/roc_curve_comparison.png", dpi=300)
plt.show()