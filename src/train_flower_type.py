import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

from dataset import FlowerTypeDataset


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


dataset = FlowerTypeDataset(
    root_dir="../data/raw",
    transform=transform
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)


model = models.resnet18(pretrained=True)

num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 5

for epoch in tqdm(range(epochs), desc="Training"):

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

    for images, labels in pbar:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    pbar.set_postfix(los=avg_loss, acc=accuracy)

    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Accuracy {accuracy:.2f}%")


torch.save(model.state_dict(), "../models/flower_type_model.pth")