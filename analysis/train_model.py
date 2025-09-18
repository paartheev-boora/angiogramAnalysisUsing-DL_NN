import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset_loader import AngioClipDataset
from models.cnn_lstm_nested import NestedCNNLSTM
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "C:/Users/arunj/Documents/Project/output_dataset_heirarchy"
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "models/nested_best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------
# Dataset and Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = AngioClipDataset(DATA_DIR, clip_length=10, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# Model, Optimizer, Loss
# -------------------------
model = NestedCNNLSTM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# -------------------------
# Training Loop
# -------------------------
best_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
    for frames, label in loop:
        frames, label = frames.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            output = model(frames)
            loss = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Saved new best model!")

print("🎉 Training Complete")
