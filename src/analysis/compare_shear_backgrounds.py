import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path("results/analysis/shear_response_comparison")
OUTPUT_DIR = DATA_DIR

# Load data
high = pd.read_csv(DATA_DIR / "high_background_shear_response.csv")
low = pd.read_csv(DATA_DIR / "low_background_shear_response.csv")

# -------- Plot Stickiness --------
plt.figure()

plt.plot(high["shear"], high["stickiness"], marker="o", label="High background")
plt.plot(low["shear"], low["stickiness"], marker="o", label="Low background")

plt.xlabel("Shear Stress")
plt.ylabel("Stickiness (Adhesion)")
plt.title("Shear vs Stickiness Comparison")
plt.legend()
plt.grid()

plt.savefig(OUTPUT_DIR / "compare_stickiness.png")
plt.close()


# -------- Plot Secretion --------
plt.figure()

plt.plot(high["shear"], high["secretion"], marker="o", label="High background")
plt.plot(low["shear"], low["secretion"], marker="o", label="Low background")

plt.xlabel("Shear Stress")
plt.ylabel("Secretion Rate")
plt.title("Shear vs Secretion Comparison")
plt.legend()
plt.grid()

plt.savefig(OUTPUT_DIR / "compare_secretion.png")
plt.close()


# -------- Plot Morphology --------
plt.figure()

plt.plot(high["shear"], high["morphology"], marker="o", label="High background")
plt.plot(low["shear"], low["morphology"], marker="o", label="Low background")

plt.xlabel("Shear Stress")
plt.ylabel("Morphology Change")
plt.title("Shear vs Morphology Comparison")
plt.legend()
plt.grid()

plt.savefig(OUTPUT_DIR / "compare_morphology.png")
plt.close()


print("Comparison plots saved in:", OUTPUT_DIR)