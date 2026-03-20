import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import FlowerTypeDataset


# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv("../data/labels.csv")
print("Total labeled images:", len(df))
print("\nClass distribution:")
print(df["flower_type"].value_counts())

# -------------------------
# Stratified split (70/15/15)
# Stratify ensures every flower type is represented
# in train, val, and test proportionally
# -------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["flower_type"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["flower_type"],
    random_state=42
)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("\nTrain class distribution:")
print(train_df["flower_type"].value_counts())

# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

# -------------------------
# Image transforms
# -------------------------
imagenet_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    imagenet_norm
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    imagenet_norm
])

# -------------------------
# Datasets
# -------------------------
train_dataset = FlowerTypeDataset(train_df, "../data/raw", train_transform)
val_dataset   = FlowerTypeDataset(val_df,   "../data/raw", eval_transform)
test_dataset  = FlowerTypeDataset(test_df,  "../data/raw", eval_transform)

print("\nClasses:", train_dataset.classes)

# -------------------------
# Weighted sampler to handle class imbalance
# This gives each class an equal chance of being
# sampled during training, regardless of its size
# -------------------------
class_counts = train_df["flower_type"].value_counts()

# Weight per class = 1 / number of samples in that class
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

# Assign a weight to every sample in the training set
sample_weights = train_df["flower_type"].map(class_weights).tolist()
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True  # Allows minority classes to be resampled
)

# -------------------------
# DataLoaders
# Note: shuffle=False when using a sampler
# -------------------------
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)

# -------------------------
# Model: ResNet18 with frozen backbone
# -------------------------
num_classes = len(train_dataset.classes)
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all layers — only train the final classification head
# This prevents overfitting on a small dataset
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -------------------------
# Loss and optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------------
# Training config
# -------------------------
num_epochs = 30
patience   = 7

best_val_accuracy = 0.0
epochs_no_improve = 0

# -------------------------
# Training loop
# -------------------------
for epoch in range(num_epochs):

    # --- Train ---
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(
            loss=f"{total_loss / len(pbar):.4f}",
            acc=f"{100 * correct / total:.1f}%"
        )

    train_accuracy = 100 * correct / total
    avg_loss       = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}%")

    # --- Validate ---
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"           | Val Acc:   {val_accuracy:.2f}%")

    # --- Early stopping and best model checkpoint ---
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), "../models/flower_type_model_best.pth")
        print(f"           | Best model saved! (val acc: {best_val_accuracy:.2f}%)")
    else:
        epochs_no_improve += 1
        print(f"           | No improvement ({epochs_no_improve}/{patience})")
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# -------------------------
# Final test evaluation (load best model)
# -------------------------
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load("../models/flower_type_model_best.pth"))
model.eval()

correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        labels  = labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

test_accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy:      {test_accuracy:.2f}%")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

# -------------------------
# Per-class breakdown
# -------------------------
from sklearn.metrics import classification_report

all_preds  = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes
))