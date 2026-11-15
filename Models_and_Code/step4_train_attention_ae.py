import os
import time
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 1. Check GPU availability (ASUS TUF NVIDIA GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on Device: {device}")

# 2. Custom Dataset Loader for Astronomical Cutouts
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

train_dataset = AstronomyDataset("../02_Datasets/train_normal", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f"Loaded {len(train_dataset)} normal galaxy images for training.")

# 3. Novel Architecture: Spatial + Channel Attention Module (CBAM-inspired)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Focus on bright celestial objects and suppress empty black sky noise
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        return x * attention_map

# 4. Attention-Enhanced Convolutional Autoencoder (Our Proposed Model)
class AttentionConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(AttentionConvolutionalAutoencoder, self).__init__()
        # Encoder: Compresses 64x64x3 image into latent feature space
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # -> 32x32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 16x16x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 8x8x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Attention Bottleneck: Directs focus to galaxy core & morphological features
        self.attention = SpatialAttention(kernel_size=7)
        
        # Decoder: Reconstructs the normal galaxy from latent representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 16x16x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 32x32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> 64x64x3
            nn.Sigmoid() # Output pixel values between [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        attended = self.attention(encoded)
        reconstructed = self.decoder(attended)
        return reconstructed

# 5. Initialize Model, Loss Function (MSE), and Optimizer
model = AttentionConvolutionalAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 6. Training Loop (50 Epochs)
epochs = 50
loss_history = []
start_time = time.time()

print("\nStarting Model Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    loss_history.append(epoch_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Reconstruction Loss (MSE): {epoch_loss:.6f}")

total_time = time.time() - start_time
print(f"\n✅ Training Completed in {total_time:.2f} seconds!")

# 7. Save Trained Model and Training Loss Curve
os.makedirs("../04_Results_and_Paper", exist_ok=True)
torch.save(model.state_dict(), "../03_Models_and_Code/attention_autoencoder_model.pth")

plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), loss_history, color='royalblue', linewidth=2)
plt.title("Training Convergence: Attention-Enhanced Convolutional Autoencoder")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE Loss)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("../04_Results_and_Paper/training_loss_curve.png", dpi=300)
plt.show()