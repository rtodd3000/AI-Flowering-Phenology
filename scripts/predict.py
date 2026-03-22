import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image


# -------------------------
# Config
# -------------------------
MODEL_PATH = "../models/flower_type_model_finetuned.pth"

CLASSES = [
    "Bombax Ceiba",
    "Lunalilo Yellow Shower Tree",
    "Queen's White Shower Tree",
    "Rainbow Shower Tree"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Confidence threshold — predictions below this are flagged as uncertain
# (useful later for pseudo-labeling)
CONFIDENCE_THRESHOLD = 0.95

# -------------------------
# Image transform (same as eval transform in training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    """Load the fine-tuned flower type model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    if not os.path.isfile(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Make sure you have run finetune_flower_type.py first.")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def predict_image(model, device, img_path):
    """
    Run inference on a single image.
    Returns the predicted class, confidence, and all class probabilities.
    """
    if not os.path.isfile(img_path):
        return None, None, None, f"File not found: {img_path}"

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, None, None, f"Could not open image: {e}"

    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs     = model(image_tensor)
        probs       = F.softmax(outputs, dim=1).squeeze()  # Convert to probabilities
        confidence  = probs.max().item()
        pred_idx    = probs.argmax().item()
        pred_class  = CLASSES[pred_idx]

    return pred_class, confidence, probs.cpu().tolist(), None


def print_prediction(img_path, pred_class, confidence, probs):
    """Print a clean prediction result for a single image."""
    print(f"\nImage: {os.path.basename(img_path)}")
    print("-" * 45)

    uncertain_flag = " ⚠️  LOW CONFIDENCE" if confidence < CONFIDENCE_THRESHOLD else ""
    print(f"Predicted: {pred_class} ({confidence*100:.1f}%){uncertain_flag}")

    print("\nAll class probabilities:")
    # Sort by probability descending
    sorted_probs = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        bar    = "█" * int(prob * 30)
        marker = " ◄" if cls == pred_class else ""
        print(f"  {cls:<35} {prob*100:5.1f}%  {bar}{marker}")


def predict_single(img_path):
    """Predict flower type for a single image."""
    model, device = load_model()
    pred_class, confidence, probs, error = predict_image(model, device, img_path)

    if error:
        print(f"Error: {error}")
        return

    print_prediction(img_path, pred_class, confidence, probs)


def predict_folder(folder_path):
    """Predict flower types for all images in a folder."""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    # Collect all image files
    image_files = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"No images found in {folder_path}")
        return

    print(f"Found {len(image_files)} images in {folder_path}")

    model, device = load_model()

    results     = []
    low_conf    = []

    for img_path in image_files:
        pred_class, confidence, probs, error = predict_image(model, device, img_path)

        if error:
            print(f"Skipping {os.path.basename(img_path)}: {error}")
            continue

        print_prediction(img_path, pred_class, confidence, probs)
        results.append((img_path, pred_class, confidence))

        if confidence < CONFIDENCE_THRESHOLD:
            low_conf.append((img_path, pred_class, confidence))

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 45)
    print(f"SUMMARY — {len(results)} images processed")
    print("=" * 45)

    # Count per class
    from collections import Counter
    class_counts = Counter(pred for _, pred, _ in results)
    print("\nPredictions per class:")
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        print(f"  {cls:<35} {count}")

    # Average confidence
    avg_conf = sum(c for _, _, c in results) / len(results)
    print(f"\nAverage confidence: {avg_conf*100:.1f}%")

    # Low confidence warnings
    if low_conf:
        print(f"\n⚠️  {len(low_conf)} image(s) below {CONFIDENCE_THRESHOLD*100:.0f}% confidence threshold:")
        for img_path, pred, conf in low_conf:
            print(f"  {os.path.basename(img_path):<40} {pred} ({conf*100:.1f}%)")
    else:
        print(f"\n✓ All predictions above {CONFIDENCE_THRESHOLD*100:.0f}% confidence threshold")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower type from image(s)")
    parser.add_argument("--image",  type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    if args.image and args.folder:
        print("Error: Please provide either --image or --folder, not both.")
        sys.exit(1)
    elif args.image:
        predict_single(args.image)
    elif args.folder:
        predict_folder(args.folder)
    else:
        parser.print_help()