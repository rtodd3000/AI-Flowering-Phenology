import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------
# Base directory
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Load data (YOUR paths)
# -------------------------
preds = pd.read_csv(os.path.join(BASE_DIR, "..", "output", "predictions.csv"))
meta  = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "labels.csv"))

# -------------------------
# Fix filename mismatch (.jpg issue)
# -------------------------
preds["image"] = preds["image"].str.replace(r"\.\w+$", "", regex=True)
meta["image_name"] = meta["image_name"].str.replace(r"\.\w+$", "", regex=True)

# -------------------------
# Merge
# -------------------------
df = preds.merge(meta, left_on="image", right_on="image_name")

print("Merged rows:", len(df))

if len(df) == 0:
    print("❌ ERROR: Merge failed — filenames don't match")
    exit()

# -------------------------
# Fix columns
# -------------------------
df["site"] = df["site_y"]
df["date"] = pd.to_datetime(df["date"])

# -------------------------
# Choose model
# -------------------------
df["intensity_used"] = df["model1_intensity"]

# Convert to numeric
intensity_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}
df["intensity_used"] = df["intensity_used"].map(intensity_map)

# -------------------------
# Group data
# -------------------------
grouped = df.groupby(
    ["site", "flower_type", "date"]
)["intensity_used"].mean().reset_index()

print("Sites found:", grouped["site"].unique())

# -------------------------
# Save plots in plot/
# -------------------------
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Plot
# -------------------------
for site in grouped["site"].unique():
    plt.figure()

    site_data = grouped[grouped["site"] == site]

    for flower in site_data["flower_type"].unique():
        flower_data = site_data[site_data["flower_type"] == flower]

        plt.plot(
            flower_data["date"],
            flower_data["intensity_used"],
            label=flower
        )

    plt.title(f"Flowering Intensity Over Time — {site}")
    plt.xlabel("Date")
    plt.ylabel("Flowering Intensity")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()

    filename = os.path.join(
        OUTPUT_DIR,
        f"{site.replace(' ', '_')}_plot.png"
    )

    plt.savefig(filename)
    plt.close()

    print(f"✅ Saved plot to: {filename}")