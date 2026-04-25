import os
import numpy as np
import onnxruntime as ort
from collections import Counter
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener() # Get Image to work on HEIF

# -------------------------
# Config
# -------------------------
FLOWER_MODEL_PATH    = "../models/flower_type_model.onnx"
INTENSITY_MODEL_PATH = "../models/intensity_model.onnx"

FLOWER_CLASSES = [
    "Bombax Ceiba",
    "Lunalilo Yellow Shower Tree",
    "Queen's White Shower Tree",
    "Rainbow Shower Tree"
]

# 4-level intensity scale
INTENSITY_CLASSES = [
    "0",
    "1",
    "2",
    "3"
]

IMAGE_EXTENSIONS = {".heif", ".heic", ".HEIF", ".HEIC"}

# -------------------------
# Image transform (must match eval transform used in training)
# -------------------------
def preprocess(image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    return img_array[np.newaxis, :].astype(np.float32)  # add batch dim

# -------------------------
# Model loading
# -------------------------

def load_both_models():
    if not os.path.isfile(FLOWER_MODEL_PATH):
        print(f"Error: Flower model not found at {FLOWER_MODEL_PATH}")
        return None, None
    if not os.path.isfile(INTENSITY_MODEL_PATH):
        print(f"Error: Intensity model not found at {INTENSITY_MODEL_PATH}")
        return None, None

    flower_session    = ort.InferenceSession(FLOWER_MODEL_PATH)
    intensity_session = ort.InferenceSession(INTENSITY_MODEL_PATH)
    return flower_session, intensity_session


# -------------------------
# Inference
# -------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_image(flower_session, intensity_session, img_path):
    """
    Run both models on a single image.
    Returns a result dict with predictions and probabilities.
    """
    date = ""
    if not os.path.isfile(img_path):
        return None, f"File not found: {img_path}"

    try:
        image = Image.open(img_path)
        exif = image.getexif()
        date = exif.get(306, "").split(" ")[0].replace(":", "-")
        image = image.convert("RGB")
    except Exception as e:
        return None, f"Could not open image: {e}"

    tensor = preprocess(image)

    # Flower prediction
    flower_output = flower_session.run(None, {"input": tensor})[0][0]
    flower_probs  = softmax(flower_output)
    flower_pred   = FLOWER_CLASSES[flower_probs.argmax()]
    flower_conf   = float(flower_probs.max())

    # Intensity prediction
    intensity_output = intensity_session.run(None, {"input": tensor})[0][0]
    intensity_probs  = softmax(intensity_output)
    intensity_pred   = INTENSITY_CLASSES[intensity_probs.argmax()]
    intensity_conf   = float(intensity_probs.max())

    result = {
        "image_path":      img_path,
        "date_created":    date,
        "flower_pred":     flower_pred,
        "flower_conf":     flower_conf,
        "flower_probs":    flower_probs.tolist(),
        "intensity_pred":  intensity_pred,
        "intensity_conf":  round(float(intensity_conf), 4),
        "intensity_probs": intensity_probs.tolist(),
    }

    return result, None

# -------------------------
# Output formatting
# -------------------------
def print_prediction(img_path, result):
    """Print a clean formatted prediction for a single image."""
    console_string = f"\n\nImage: {os.path.basename(img_path)}"
    console_string += "\n" + "=" * 50
    console_string += f"\nFlower Type:  {result['flower_pred']} ({result['flower_conf']*100:.1f}%)"
    console_string += f"\nIntensity:    {result['intensity_pred']} ({result['intensity_conf']*100:.1f}%)"

    # Flower type probabilities
    console_string += "\n\nAll flower type probabilities:"
    sorted_flower = sorted(zip(FLOWER_CLASSES, result["flower_probs"]), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_flower:
        bar    = "█" * int(prob * 30)
        marker = " ◄" if cls == result["flower_pred"] else ""
        console_string += f"\n  {cls:<35} {prob*100:5.1f}%  {bar}{marker}"

    # Intensity probabilities
    console_string += "\n\nAll intensity probabilities:"
    sorted_intensity = sorted(zip(INTENSITY_CLASSES, result["intensity_probs"]), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_intensity:
        bar    = "█" * int(prob * 30)
        marker = " ◄" if cls == result["intensity_pred"] else ""
        console_string += f"\n  {cls:<35} {prob*100:5.1f}%  {bar}{marker}"

    console_string += "\n" + "-" * 50

    return console_string


# -------------------------
# Single image prediction
# -------------------------
def predict_single(img_path):
    """Predict flower type and intensity for a single image."""    
    flower_session, intensity_session = load_both_models()
    if not flower_session:
        return None, "Error loading models"

    result, error = predict_image(flower_session, intensity_session, img_path)
    if error:
        return None, error

    console_string = print_prediction(img_path, result)
    console_string += "\nSingle Image Predicted"
    return result, console_string

# -------------------------
# Folder prediction
# -------------------------
def predict_folder(folder_path):
    """Predict flower type and intensity for all new images in a folder."""
    console_string = ""

    image_files = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    # Infer site from folder name
    site = os.path.basename(os.path.normpath(folder_path))

    flower_session, intensity_session = load_both_models()
    if not flower_session:
        return [], "Error loading models"

    results = []
    skipped = 0

    for img_path in image_files:   
        result, error = predict_image(flower_session, intensity_session, img_path)
        if error:
            continue
        console_string += print_prediction(img_path, result)
        results.append(result)
    # -------------------------
    # Summary
    # -------------------------
    console_string += "\n" + "=" * 50
    console_string += f"\nSUMMARY"
    console_string += "\n" + "=" * 50
    console_string += f"\nSite:                  {site}"
    console_string += f"\nTotal images found:    {len(image_files)}"
    console_string += f"\nAlready in CSV:        {skipped}"
    console_string += f"\nNewly processed:       {len(results)}"

    if results:
        # Flower type counts
        flower_counts = Counter(r["flower_pred"] for r in results)
        console_string += "\n\nFlower type predictions:"
        for cls in FLOWER_CLASSES:
            console_string += f"\n  {cls:<35} {flower_counts.get(cls, 0)}"

        # Intensity counts
        intensity_counts = Counter(r["intensity_pred"] for r in results)
        console_string += "\n\nIntensity predictions:"
        for cls in INTENSITY_CLASSES:
            console_string += f"\n  {cls:<35} {intensity_counts.get(cls, 0)}"

        # Average confidences
        avg_flower_conf    = sum(r["flower_conf"]    for r in results) / len(results)
        avg_intensity_conf = sum(r["intensity_conf"] for r in results) / len(results)
        console_string += f"\n\nAverage flower type confidence: {avg_flower_conf*100:.1f}%"
        console_string += f"\nAverage intensity confidence:   {avg_intensity_conf*100:.1f}%"
    else:
        console_string += "\nNo new images to process."

    return results, console_string