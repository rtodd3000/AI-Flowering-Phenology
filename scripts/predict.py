import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
import pandas as pd

# -------------------------
# Paths (FIXED)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_1 = os.path.join(BASE_DIR, "..", "models", "flower_type_model_finetuned.pth")
MODEL_PATH_2 = os.path.join(BASE_DIR, "..", "models", "flower_type_model_best.pth")

OUTPUT_PATH = os.path.join(BASE_DIR, "..", "output", "predictions.csv")

# -------------------------
# Config
# -------------------------
CLASSES = [
    "Bombax Ceiba",
    "Lunalilo Yellow Shower Tree",
    "Queen's White Shower Tree",
    "Rainbow Shower Tree"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# Load model
# -------------------------
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device

# -------------------------
# Predict one image
# -------------------------
def predict_image(model, device, img_path):
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).squeeze()
        confidence = probs.max().item()
        pred_idx = probs.argmax().item()
        pred_class = CLASSES[pred_idx]

    return pred_class, confidence, probs.cpu().tolist()

# -------------------------
# Convert confidence → intensity
# -------------------------
def get_intensity(conf):
    if conf > 0.85:
        return "High"
    elif conf > 0.60:
        return "Medium"
    else:
        return "Low"

# -------------------------
# Predict folder
# -------------------------
def predict_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    site_name = os.path.basename(folder_path)

    image_files = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"No images found in {folder_path}")
        return

    print(f"Found {len(image_files)} images in {folder_path}")

    # Load BOTH models
    model1, device = load_model(MODEL_PATH_1)
    model2, _ = load_model(MODEL_PATH_2)

    rows = []

    for img_path in image_files:
        try:
            pred1, conf1, _ = predict_image(model1, device, img_path)
            pred2, conf2, _ = predict_image(model2, device, img_path)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        print(f"{os.path.basename(img_path)} → {pred1} ({conf1:.2f}), {pred2} ({conf2:.2f})")

        row = {
            "image": os.path.basename(img_path),
            "site": site_name,

            "model1_class": pred1,
            "model1_conf": conf1,
            "model1_intensity": get_intensity(conf1),

            "model2_class": pred2,
            "model2_conf": conf2,
            "model2_intensity": get_intensity(conf2),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # -------------------------
    # Append to CSV (NOT overwrite)
    # -------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n💾 Saved to {OUTPUT_PATH}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower type from folder")
    parser.add_argument("--folder", type=str, help="Path to image folder")
    args = parser.parse_args()

    if args.folder:
        predict_folder(args.folder)
    else:
        parser.print_help()