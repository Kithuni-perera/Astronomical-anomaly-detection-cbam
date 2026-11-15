import os
import time
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Running Anomaly Detection Inference on: {device}")

# 2. Define the exact Model Architecture
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        return x * attention_map

class AttentionConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(AttentionConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.attention = SpatialAttention(kernel_size=7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        attended = self.attention(encoded)
        reconstructed = self.decoder(attended)
        return reconstructed

# 3. Load Trained Weights
model = AttentionConvolutionalAutoencoder().to(device)
model.load_state_dict(torch.load("../03_Models_and_Code/attention_autoencoder_model.pth", map_location=device))
model.eval()

# 4. Dataset Loader
class AstronomyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

normal_dataset = AstronomyDataset("../02_Datasets/train_normal", transform=transform)
test_dataset = AstronomyDataset("../02_Datasets/test_mixed", transform=transform)

# 5. Compute Anomaly Scores & Measure Inference Speed
def evaluate_anomalies(dataset):
    results = []
    start_time = time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            img_tensor, filename = dataset[i]
            input_batch = img_tensor.unsqueeze(0).to(device)
            reconstructed = model(input_batch)
            
            # Mean Squared Error per image as Anomaly Score
            mse = torch.mean((input_batch - reconstructed) ** 2).item()
            results.append({
                "filename": filename,
                "anomaly_score": mse,
                "input_img": img_tensor.permute(1, 2, 0).numpy(),
                "recon_img": reconstructed.squeeze(0).cpu().permute(1, 2, 0).numpy()
            })
    elapsed = time.time() - start_time
    return results, elapsed

normal_results, norm_time = evaluate_anomalies(normal_dataset)
test_results, test_time = evaluate_anomalies(test_dataset)

total_images = len(normal_dataset) + len(test_dataset)
total_inference_time = norm_time + test_time
ms_per_image = (total_inference_time / total_images) * 1000

print(f"\n⚡ BENCHMARK RESULTS:")
print(f"Total Images Evaluated: {total_images}")
print(f"Total Inference Time: {total_inference_time:.4f} seconds ({ms_per_image:.2f} ms per image)")

# 6. Rank Test Objects by Highest Anomaly Score
test_results_sorted = sorted(test_results, key=lambda x: x["anomaly_score"], reverse=True)
norm_scores = [r["anomaly_score"] for r in normal_results]
test_scores = [r["anomaly_score"] for r in test_results]

# Save top discovered anomalies to CSV linked with SDSS catalog
test_catalog = pd.read_csv("../02_Datasets/test_anomalies_catalog.csv")
top_records = []
for rank, item in enumerate(test_results_sorted[:15], 1):
    obj_id = int(item["filename"].replace("test_", "").replace(".jpg", ""))
    match = test_catalog[test_catalog["objid"] == obj_id]
    ra = match["ra"].values[0] if not match.empty else None
    dec = match["dec"].values[0] if not match.empty else None
    obj_class = match["class"].values[0] if not match.empty else "Unknown"
    redshift = match["z"].values[0] if not match.empty else None
    top_records.append({
        "Rank": rank,
        "SDSS_ObjID": obj_id,
        "RA": ra,
        "Dec": dec,
        "Spectroscopic_Class": obj_class,
        "Redshift_z": redshift,
        "Anomaly_Score": round(item["anomaly_score"], 6)
    })

top_df = pd.DataFrame(top_records)
top_df.to_csv("../04_Results_and_Paper/top_discovered_anomalies.csv", index=False)
print("\n🏆 TOP 5 DISCOVERED ANOMALOUS CELESTIAL OBJECTS:")
print(top_df.head(5).to_string(index=False))

# 7. Plot 1: Anomaly Score Distribution Histogram (Normal vs Anomalies)
plt.figure(figsize=(8, 5))
plt.hist(norm_scores, bins=30, alpha=0.7, label="Normal Galaxies (Training Distribution)", color="steelblue", density=True)
plt.hist(test_scores, bins=30, alpha=0.7, label="Test Candidates (Quasars / Outliers)", color="crimson", density=True)
threshold_95 = np.percentile(norm_scores, 95)
plt.axvline(threshold_95, color="black", linestyle="--", linewidth=1.5, label=f"95% Anomaly Threshold ({threshold_95:.4f})")
plt.title("Anomaly Score Distribution: Normal Galaxies vs. Outlier Candidates")
plt.xlabel("Reconstruction Anomaly Score (MSE)")
plt.ylabel("Normalized Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("../04_Results_and_Paper/anomaly_score_distribution.png", dpi=300)

# 8. Plot 2: Visual Comparison (Input vs Reconstructed vs Residual Map) for Top 4 Anomalies
fig, axes = plt.subplots(4, 3, figsize=(9, 11))
fig.suptitle("Top 4 Detected Astronomical Anomalies (Input vs. Reconstruction vs. Residual)", fontsize=12, fontweight="bold")

for row_idx in range(4):
    item = test_results_sorted[row_idx]
    inp = item["input_img"]
    rec = item["recon_img"]
    residual = np.abs(inp - rec).mean(axis=2)  # Pixel-wise residual error

    axes[row_idx, 0].imshow(inp)
    axes[row_idx, 0].set_title(f"Input (Rank #{row_idx+1})\nScore: {item['anomaly_score']:.4f}", fontsize=9)
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(rec)
    axes[row_idx, 1].set_title("AI Reconstructed", fontsize=9)
    axes[row_idx, 1].axis("off")

    im = axes[row_idx, 2].imshow(residual, cmap="hot", vmin=0, vmax=0.4)
    axes[row_idx, 2].set_title("Residual Anomaly Map", fontsize=9)
    axes[row_idx, 2].axis("off")

plt.tight_layout()
plt.savefig("../04_Results_and_Paper/top_anomalies_visualization.png", dpi=300)
plt.show()