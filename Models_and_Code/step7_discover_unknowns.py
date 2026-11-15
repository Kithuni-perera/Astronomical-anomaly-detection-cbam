import os
import io
import time
import glob
import requests
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔭 Initiating Blind Sky Survey for Unclassified Celestial Objects on: {device}")

# 1. Query SDSS DR17 for UNCLASSIFIED / UNLABELED Sky Objects (No Spectroscopic Confirmation Yet!)
unknown_dir = "../02_Datasets/unclassified_sky_survey"
os.makedirs(unknown_dir, exist_ok=True)

def query_unclassified_objects():
    url = "http://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
    # Select objects with NO spectroscopic classification (s.specobjid IS NULL)
    # and unusual color gradients or disturbed/blended flags
    sql = """
    SELECT TOP 250
        p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.petroR90_r
    FROM PhotoObj AS p
    LEFT OUTER JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE 
        s.specobjid IS NULL
        AND p.type = 3
        AND p.r BETWEEN 16.0 AND 18.8
        AND p.petroR90_r > 4.0
        AND (p.g - p.r > 1.2 OR p.u - p.g < 0.5)
    """
    resp = requests.get(url, params={"cmd": sql, "format": "csv"}, timeout=30)
    resp.raise_for_status()
    lines = [line for line in resp.text.splitlines() if not line.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)))

print("Querying SDSS DR17 for unstudied, unlabeled sky objects...")
unknown_df = query_unclassified_objects()
unknown_df.to_csv("../02_Datasets/unclassified_candidates_catalog.csv", index=False)
print(f"✅ Found {len(unknown_df)} unclassified astronomical candidates.")

# 2. Download 64x64 Cutouts for Blind Discovery (< 1.5 MB data)
base_img_url = "http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
downloaded_records = []

for idx, row in unknown_df.iterrows():
    params = {"ra": row["ra"], "dec": row["dec"], "scale": 0.396, "width": 64, "height": 64}
    try:
        r = requests.get(base_img_url, params=params, timeout=10)
        if r.status_code == 200:
            fname = f"unk_{int(row['objid'])}.jpg"
            fpath = os.path.join(unknown_dir, fname)
            with open(fpath, "wb") as f:
                f.write(r.content)
            downloaded_records.append((fname, row))
    except Exception:
        pass
    time.sleep(0.04)

print(f"✅ Downloaded {len(downloaded_records)} unclassified sky images for AI scanning.")

# 3. Define Attention-Enhanced CAE (CBAM)
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

# 4. Train on Normal Galaxies & Fit Latent Boundary
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

class SimpleFolderDataset(Dataset):
    def __init__(self, folder):
        self.paths = glob.glob(os.path.join(folder, "*.jpg"))
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return transform(Image.open(self.paths[idx]).convert("RGB"))

train_loader = DataLoader(SimpleFolderDataset("../02_Datasets/train_normal"), batch_size=32, shuffle=True)
model = AttentionCAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

print("Training Attention-CAE on normal galaxy morphology...")
for _ in range(35):
    model.train()
    for imgs in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        rec, _ = model(imgs)
        loss = criterion(rec, imgs)
        loss.backward()
        optimizer.step()

model.eval()
train_latents = []
with torch.no_grad():
    for imgs in train_loader:
        _, z = model(imgs.to(device))
        train_latents.append(z.cpu().numpy())

iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso_forest.fit(np.vstack(train_latents))

# 5. Scan Unclassified Objects & Rank Novel Discoveries
discoveries = []
with torch.no_grad():
    for fname, row in downloaded_records:
        fpath = os.path.join(unknown_dir, fname)
        img_pil = Image.open(fpath).convert("RGB")
        inp = transform(img_pil).unsqueeze(0).to(device)
        _, z = model(inp)
        score = -iso_forest.decision_function(z.cpu().numpy())[0]
        discoveries.append({
            "SDSS_ObjID": int(row["objid"]),
            "RA": round(row["ra"], 6),
            "Dec": round(row["dec"], 6),
            "r_mag": round(row["r"], 2),
            "g_minus_r": round(row["g"] - row["r"], 2),
            "Anomaly_Score": round(score, 5),
            "SkyServer_Link": f"http://skyserver.sdss.org/dr17/en/tools/explore/Summary.aspx?id={int(row['objid'])}",
            "img": np.array(img_pil)
        })

discoveries_sorted = sorted(discoveries, key=lambda x: x["Anomaly_Score"], reverse=True)
disc_df = pd.DataFrame([{k: v for k, v in d.items() if k != "img"} for d in discoveries_sorted[:15]])
disc_df.to_csv("../04_Results_and_Paper/novel_unclassified_discoveries.csv", index=False)

print("\n🌌 TOP 6 NEWLY DISCOVERED UNCLASSIFIED CELESTIAL CANDIDATES:")
print(disc_df.head(6)[["SDSS_ObjID", "RA", "Dec", "r_mag", "g_minus_r", "Anomaly_Score"]].to_string(index=False))

# 6. Visualize the Top 6 Novel Discoveries
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
fig.suptitle("Novel Unclassified Celestial Candidates Discovered by Attention-CAE", fontsize=12, fontweight="bold")

for idx in range(6):
    ax = axes[idx // 3, idx % 3]
    item = discoveries_sorted[idx]
    ax.imshow(item["img"])
    ax.set_title(f"Candidate #{idx+1} (Score: {item['Anomaly_Score']:.4f})\nRA: {item['RA']}, Dec: {item['Dec']}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("../04_Results_and_Paper/novel_discoveries_gallery.png", dpi=300)
plt.show()