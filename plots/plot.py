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
# Convert date
# -------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# -------------------------
# Use labeled intensity
# -------------------------
df["intensity_used"] = df["intensity"]

# -------------------------
# Group by site + date
# -------------------------
grouped = df.groupby(["site", "date"])["intensity_used"].mean().reset_index()

print("Sites found:", grouped["site"].unique())

# -------------------------
# Output directory
# -------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "plot-images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Plot per site
# -------------------------
for site in grouped["site"].unique():
    plt.figure(figsize=(20, 7))

    site_data = grouped[grouped["site"] == site].sort_values("date")

    plt.plot(
        site_data["date"],
        site_data["intensity_used"],
    )

    plt.title(f"Flowering Intensity Over Time — {site}")
    plt.xlabel("Date")
    plt.ylabel("Flowering Intensity")

    ax = plt.gca()

    # Monthly ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    def format_date(x, pos=None):
        d = mdates.num2date(x)
        return d.strftime('%b\n%Y')

    ax.xaxis.set_major_formatter(format_date)

    # Base tick style
    ax.tick_params(axis='x', length=3, width=0.5)
    plt.xticks(fontsize=7)

    # Light grid
    ax.grid(axis='x', linestyle='-', linewidth=0.3, alpha=0.3)

    # Get rid of extra space on the left and right
    ax.margins(x=0)

    plt.yticks([0, 1, 2, 3])
    plt.tight_layout()

    # Save
    filename = f"{site.replace(' ', '_')}_plot.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(filepath)
    plt.close()