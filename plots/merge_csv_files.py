import pandas as pd
import glob

# -------------------------
# Combine intensity
# -------------------------
intensity_files = glob.glob("output/*_intensity_predictions.csv")
intensity_df = pd.concat([pd.read_csv(f) for f in intensity_files])

intensity_df.to_csv("output/all_intensity.csv", index=False)
print("Combined intensity CSV saved")

# -------------------------
# Load flower predictions
# -------------------------
flower_df = pd.read_csv("output/predictions.csv")

# -------------------------
# CLEAN KEYS (important)
# -------------------------
intensity_df["image"] = intensity_df["image"].str.strip()
flower_df["image"] = flower_df["image"].str.strip()

# (optional fix if extensions mismatch)
# intensity_df["image"] = intensity_df["image"].str.replace(".jpg", "", regex=False)
# flower_df["image"] = flower_df["image"].str.replace(".jpg", "", regex=False)

# -------------------------
# DEBUG
# -------------------------
print("Intensity columns:", intensity_df.columns)
print("Flower columns:", flower_df.columns)

# -------------------------
# Merge
# -------------------------
df = pd.merge(
    intensity_df,
    flower_df,
    on=["image", "site"]  # must exist in BOTH
)

df.to_csv("output/all_predictions.csv", index=False)

print("Merged dataset saved")
print("Rows:", len(df))