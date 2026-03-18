import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import FlowerTypeDataset
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------
# Load CSV and site info
# -------------------------
df = pd.read_csv("../data/labels.csv")
print("All sites:", df["site"].unique())

# First split: train (70%) and temp (30%)
'''train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["flower_type"],
    random_state=42
)

# Second split: validation (15%) and test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["flower_type"],
    random_state=42
)
'''

train_sites = ["Bombax_SE", "East-West-Maile", "Maile East End"]
val_sites = ["Maile Library"]
test_sites = ["Maile West Corner"]

train_df = df[df["site"].isin(train_sites)]
val_df = df[df["site"].isin(val_sites)]
test_df = df[df["site"].isin(test_sites)]

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Image transforms
# -------------------------
'''transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
'''
# Train transform (augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # 🔥 KEY ADD
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor()
])

# Validation / Test transform (clean)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# Dataset class fix for .jpg
# -------------------------
class FixedFlowerTypeDataset(FlowerTypeDataset):
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_name"]
        label = self.df.iloc[idx]["flower_type"]
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]["site"], img_name)

        # Check if file exists; if not, try adding .jpg
        if not os.path.isfile(img_path):
            img_path_jpg = img_path + ".jpg"
            if os.path.isfile(img_path_jpg):
                img_path = img_path_jpg
            else:
                raise FileNotFoundError(f"File not found: {img_path} or {img_path_jpg}")

        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Map class to index
        label_idx = self.classes.index(label)

        return image, label_idx

# -------------------------
# Load datasets
# -------------------------
dataset = FixedFlowerTypeDataset(df, "../data/raw", test_transform)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

train_dataset = FixedFlowerTypeDataset(train_df, "../data/raw", train_transform)
val_dataset = FixedFlowerTypeDataset(val_df, "../data/raw", test_transform)
test_dataset = FixedFlowerTypeDataset(test_df, "../data/raw", test_transform)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# -------------------------
# Dataloaders
# -------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# -------------------------
# Model
# -------------------------
num_classes = len(dataset.classes)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -------------------------
# Loss + optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -------------------------
# Training
# -------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / (total / labels.size(0))
        accuracy = 100 * correct / total
        pbar.set_postfix(loss=avg_loss, acc=accuracy)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# -------------------------
# Final Test Evaluation
# -------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), "../models/flower_type_model.pth")
print("Model saved!")