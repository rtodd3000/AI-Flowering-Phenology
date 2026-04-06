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
# Config
# -------------------------
MODEL_PATH = "models/intensity_model_finetuned.pth"
CLASSES = ["0", "1", "2", "3"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
CONFIDENCE_THRESHOLD = 0.95

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
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    if not os.path.isfile(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    return model, device

# -------------------------
# Predict single image
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

    return pred_class, confidence

# -------------------------
# Predict ONE folder
# -------------------------
def predict_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return

    image_files = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"No images found in {folder_path}")
        return

    print(f"\nProcessing {len(image_files)} images in {folder_path}")

    model, device = load_model()

    results = []
    site = os.path.basename(folder_path)

    for img_path in image_files:
        filename = os.path.basename(img_path)

        # ⚠️ adjust if your filename format is different
        date = filename.split("_")[0]

        try:
            pred_class, confidence = predict_image(model, device, img_path)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        print(f"{filename} → {pred_class} ({confidence*100:.1f}%)")

        results.append({
            "image": filename,
            "date": date,
            "site": site,
            "intensity": int(pred_class),
            "confidence": confidence
        })

    if len(results) == 0:
        print("No results to save.")
        return

    os.makedirs("output", exist_ok=True)

    df = pd.DataFrame(results)
    output_file = f"output/{site}_intensity_predictions.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Saved CSV to: {output_file}")
    print(f"Rows saved: {len(df)}")

# -------------------------
# Predict ALL sites
# -------------------------
def predict_all_sites(root_dir):
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found: {root_dir}")
        return

    folders = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    print(f"\nFound {len(folders)} site folders")

    for folder in folders:
        print("\n" + "=" * 60)
        print(f"Processing site: {os.path.basename(folder)}")
        print("=" * 60)
        predict_folder(folder)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Path to one folder")
    parser.add_argument("--all", action="store_true", help="Run on all sites in data/raw")

    args = parser.parse_args()

    if args.folder:
        predict_folder(args.folder)
    elif args.all:
        predict_all_sites("data/raw")
    else:
        print("Use --folder or --all")