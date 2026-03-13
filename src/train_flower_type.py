import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import FlowerTypeDataset


# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Image transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -------------------------
# Load dataset
# -------------------------
dataset = FlowerTypeDataset(
    root_dir="../data/raw",
    transform=transform
)

print("Total images:", len(dataset))


# -------------------------
# Train / Val / Test split
# -------------------------
dataset_size = len(dataset)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

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

optimizer = optim.Adam(model.parameters(), lr=0.0005)


# -------------------------
# Training
# -------------------------
num_epochs = 5

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
torch.save(model.state_dict(), "../models/flower_model.pth")

print("Model saved!")