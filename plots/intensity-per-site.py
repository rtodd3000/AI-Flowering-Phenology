import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------
# Base directory
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Load data
# -------------------------
intensity = pd.read_csv(os.path.join(BASE_DIR, "..", "output", "all_intensity.csv"))
labels = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "labels.csv"))

# -------------------------
# Clean filenames (IMPORTANT)
# -------------------------
intensity["image"] = intensity["image"].str.replace(r"\.\w+$", "", regex=True).str.strip()
labels["image_name"] = labels["image_name"].str.replace(r"\.\w+$", "", regex=True).str.strip()

# -------------------------
# Merge to get DATE
# -------------------------
df = intensity.merge(
    labels[["image_name", "site", "date"]],
    left_on=["image", "site"],
    right_on=["image_name", "site"],
    how="left"
)

# -------------------------
# Check merge success
# -------------------------
print("Columns after merge:", df.columns)
print("Merged rows:", len(df))

if len(df) == 0:
    print("❌ ERROR: Merge failed — check image/site matching")
    exit()

# -------------------------
# Ensure correct date column
# -------------------------
if "date_y" in df.columns:
    df = df.rename(columns={"date_y": "date"})

# -------------------------
# Convert date
# -------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Drop rows where date failed
df = df.dropna(subset=["date"])

# -------------------------
# Use intensity directly
# -------------------------
df["intensity_used"] = df["intensity"]

# -------------------------
# Group by site + date
# -------------------------
grouped = df.groupby(
    ["site", "date"]
)["intensity_used"].mean().reset_index()

print("Sites found:", grouped["site"].unique())

# -------------------------
# Output directory
# -------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Plot per site
# -------------------------
for site in grouped["site"].unique():
    plt.figure()

    site_data = grouped[grouped["site"] == site].sort_values("date")

    # Smooth it
    site_data["smoothed"] = site_data["intensity_used"].rolling(window=5, min_periods=1).mean()

    plt.plot(
        site_data["date"],
        site_data["smoothed"],
        marker="o"
    )

    plt.title(f"Flowering Intensity Over Time — {site}")
    plt.xlabel("Date")
    plt.ylabel("Flowering Intensity")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()