import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# -------------------------
# Base directory
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Load labeled data
# -------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "labels.csv"))

# -------------------------
# Check columns (optional)
# -------------------------
print("Columns:", df.columns)

# -------------------------
# Convert date
# -------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# -------------------------
# Use labeled intensity
# -------------------------
df["intensity_used"] = df["intensity"]   # change if column name differs

# -------------------------
# Group by site + date
# -------------------------
grouped = df.groupby(["site", "date"])["intensity_used"].mean().reset_index()

print("Sites found:", grouped["site"].unique())

# -------------------------
# Output directory (optional)
# -------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Plot per site
# -------------------------
for site in grouped["site"].unique():
    plt.figure(figsize=(14, 7))

    site_data = grouped[grouped["site"] == site].sort_values("date")

    # Smooth line
    site_data["smoothed"] = site_data["intensity_used"].rolling(
        window=5, min_periods=1
    ).mean()

    # Raw
    plt.plot(
        site_data["date"],
        site_data["intensity_used"],
        label="Raw Intensity"
    )

    # Smoothed
    plt.plot(
        site_data["date"],
        site_data["smoothed"],
        linewidth=2,
        label="Smoothed Intensity"
    )

    plt.title(f"Flowering Intensity Over Time — {site}")
    plt.xlabel("Date")
    plt.ylabel("Flowering Intensity")
    plt.legend()

    # Clean x-axis (years + months)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=30)
    plt.yticks([0, 1, 2, 3])

    plt.tight_layout()
    plt.show()