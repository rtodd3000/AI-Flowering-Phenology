import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

model = resnet18()
model.fc = torch.nn.Linear(512, num_classes)
model.load_state_dict(torch.load("flower_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open("test_flower.jpg").convert("RGB")
image = transform(image).unsqueeze(0)

output = model(image)
_, predicted = output.max(1)

print("Predicted class:", predicted.item())