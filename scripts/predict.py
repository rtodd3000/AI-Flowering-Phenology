import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
from collections import Counter
import pandas as pd


# Config
FLOWER_MODEL_PATH    = "../models/flower_type_model_finetuned.pth"
INTENSITY_MODEL_PATH = "../models/intensity_model_finetuned.pth"
CSV_PATH             = "../data/labels.csv"

FLOWER_CLASSES = [
    "Bombax Ceiba",
    "Lunalilo Yellow Shower Tree",
    "Queen's White Shower Tree",
    "Rainbow Shower Tree"
]

# 4-level intensity scale
INTENSITY_CLASSES = [
    "0 - No Flowers",
    "1 - Few Bunches",
    "2 - Transition",
    "3 - Full Bloom"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Image transform (must match eval transform used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# CSV helpers
def load_csv():
    """Load the labels CSV and return as a DataFrame."""
    if os.path.isfile(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    else:
        print(f"Warning: CSV not found at {CSV_PATH}, a new one will be created.")
        return pd.DataFrame(columns=["image_name", "site", "date", "intensity", "flower_type"])


def already_in_csv(df, image_name, site):
    """Check if an image from a given site is already in the CSV."""
    return ((df["image_name"] == image_name) & (df["site"] == site)).any()


def append_to_csv(df, image_name, site, flower_type, intensity):
    """
    Append a new row to the DataFrame.
    Site is inferred from folder name.
    Date is left blank for manual entry.
    Intensity is stored as raw integer (0, 1, 2, 3).
    """
    # Extract raw intensity number from label string e.g. "2 - Transition" -> 2
    intensity_int = int(intensity.split(" - ")[0])

    new_row = pd.DataFrame([{
        "image_name":  image_name,
        "site":        site,
        "date":        "",
        "intensity":   intensity_int,
        "flower_type": flower_type
    }])

    return pd.concat([df, new_row], ignore_index=True)


def save_csv(df):
    """Save the DataFrame back to CSV."""
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✓ CSV updated: {CSV_PATH}")


# Model loading
def load_model(model_path, num_classes, model_name):
    """Load a saved ResNet18 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(model_path):
        print(f"Error: {model_name} model not found at {model_path}")
        print("Make sure you have run the fine-tuning script first.")
        sys.exit(1)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def load_both_models():
    """Load flower type and intensity models."""
    flower_model,    device = load_model(FLOWER_MODEL_PATH,    len(FLOWER_CLASSES),    "Flower type")
    intensity_model, _      = load_model(INTENSITY_MODEL_PATH, len(INTENSITY_CLASSES), "Intensity")
    intensity_model.to(device)
    return flower_model, intensity_model, device


# Inference
def predict_image(flower_model, intensity_model, device, img_path):
    """
    Run both models on a single image.
    Returns a result dict with predictions and probabilities.
    """
    if not os.path.isfile(img_path):
        return None, f"File not found: {img_path}"

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, f"Could not open image: {e}"

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Flower type prediction
        flower_probs    = F.softmax(flower_model(image_tensor),    dim=1).squeeze()
        flower_conf     = flower_probs.max().item()
        flower_pred     = FLOWER_CLASSES[flower_probs.argmax().item()]

        # Intensity prediction
        intensity_probs = F.softmax(intensity_model(image_tensor), dim=1).squeeze()
        intensity_conf  = intensity_probs.max().item()
        intensity_pred  = INTENSITY_CLASSES[intensity_probs.argmax().item()]

    result = {
        "flower_pred":     flower_pred,
        "flower_conf":     flower_conf,
        "flower_probs":    flower_probs.cpu().tolist(),
        "intensity_pred":  intensity_pred,
        "intensity_conf":  intensity_conf,
        "intensity_probs": intensity_probs.cpu().tolist(),
    }

    return result, None


# Output formatting
def print_prediction(img_path, result):
    """Print a clean formatted prediction for a single image."""
    print(f"\nImage: {os.path.basename(img_path)}")
    print("=" * 50)
    print(f"Flower Type:  {result['flower_pred']} ({result['flower_conf']*100:.1f}%)")
    print(f"Intensity:    {result['intensity_pred']} ({result['intensity_conf']*100:.1f}%)")

    # Flower type probabilities
    print("\nAll flower type probabilities:")
    sorted_flower = sorted(zip(FLOWER_CLASSES, result["flower_probs"]), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_flower:
        bar    = "█" * int(prob * 30)
        marker = " ◄" if cls == result["flower_pred"] else ""
        print(f"  {cls:<35} {prob*100:5.1f}%  {bar}{marker}")

    # Intensity probabilities
    print("\nAll intensity probabilities:")
    sorted_intensity = sorted(zip(INTENSITY_CLASSES, result["intensity_probs"]), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_intensity:
        bar    = "█" * int(prob * 30)
        marker = " ◄" if cls == result["intensity_pred"] else ""
        print(f"  {cls:<35} {prob*100:5.1f}%  {bar}{marker}")

    print("-" * 50)


# Single image prediction
def predict_single(img_path, site_override=None):
    """Predict flower type and intensity for a single image."""
    flower_model, intensity_model, device = load_both_models()
    df = load_csv()

    image_name = os.path.basename(img_path)
    site       = site_override if site_override else os.path.basename(os.path.dirname(img_path))

    if already_in_csv(df, image_name, site):
        print(f"Skipping {image_name} — already in CSV.")
        return

    result, error = predict_image(flower_model, intensity_model, device, img_path)
    if error:
        print(f"Error: {error}")
        return

    print_prediction(img_path, result)

    df = append_to_csv(df, image_name, site, result["flower_pred"], result["intensity_pred"])
    save_csv(df)
    print(f"Added: {image_name} | Site: {site} | {result['flower_pred']} | {result['intensity_pred']}")


# Folder prediction
def predict_folder(folder_path):
    """Predict flower type and intensity for all new images in a folder."""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    image_files = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"No images found in {folder_path}")
        return

    # Infer site from folder name
    site = os.path.basename(os.path.normpath(folder_path))

    df = load_csv()
    flower_model, intensity_model, device = load_both_models()

    results = []
    skipped = 0

    for img_path in image_files:
        image_name = os.path.basename(img_path)

        # Skip images already in CSV
        if already_in_csv(df, image_name, site):
            skipped += 1
            continue

        result, error = predict_image(flower_model, intensity_model, device, img_path)
        if error:
            print(f"Skipping {image_name}: {error}")
            continue

        print_prediction(img_path, result)
        df = append_to_csv(df, image_name, site, result["flower_pred"], result["intensity_pred"])
        results.append((img_path, result))

    # Save all new rows at once
    if results:
        save_csv(df)

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 50)
    print(f"SUMMARY")
    print("=" * 50)
    print(f"Site:                  {site}")
    print(f"Total images found:    {len(image_files)}")
    print(f"Already in CSV:        {skipped}")
    print(f"Newly processed:       {len(results)}")

    if results:
        # Flower type counts
        flower_counts = Counter(r["flower_pred"] for _, r in results)
        print("\nFlower type predictions:")
        for cls in FLOWER_CLASSES:
            print(f"  {cls:<35} {flower_counts.get(cls, 0)}")

        # Intensity counts
        intensity_counts = Counter(r["intensity_pred"] for _, r in results)
        print("\nIntensity predictions:")
        for cls in INTENSITY_CLASSES:
            print(f"  {cls:<35} {intensity_counts.get(cls, 0)}")

        # Average confidences
        avg_flower_conf    = sum(r["flower_conf"]    for _, r in results) / len(results)
        avg_intensity_conf = sum(r["intensity_conf"] for _, r in results) / len(results)
        print(f"\nAverage flower type confidence: {avg_flower_conf*100:.1f}%")
        print(f"Average intensity confidence:   {avg_intensity_conf*100:.1f}%")
    else:
        print("\nNo new images to process.")


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict flower type and intensity, and save results to labels.csv"
    )
    parser.add_argument("--image",  type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--site",   type=str, help="Site name (optional, inferred from folder if not provided)")
    args = parser.parse_args()

    if args.image and args.folder:
        print("Error: Please provide either --image or --folder, not both.")
        sys.exit(1)
    elif args.image:
        predict_single(args.image, site_override=args.site)
    elif args.folder:
        predict_folder(args.folder)
    else:
        parser.print_help()