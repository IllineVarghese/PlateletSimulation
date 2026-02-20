import numpy as np
import matplotlib.pyplot as plt

# ---- load data ----
P = np.load("results/week3/positions_saved.npy")
A = np.load("results/week3/activation_saved.npy")

# last frame
pos = P[-1]
act = A[-1]

x = pos[:, 0]
y = pos[:, 1]

# ---- plot ----
plt.figure(figsize=(6, 6))

# activation scatter
sc = plt.scatter(x, y, c=act, cmap="hot", s=8, vmin=0, vmax=1)
plt.colorbar(sc, label="Activation")

# draw vessel wall (set this to your cylinder radius; 1.0 in your simulation)
theta = np.linspace(0, 2 * np.pi, 200)
R = 1.0
plt.plot(R * np.cos(theta), R * np.sin(theta), linewidth=1)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Week 3: Near-Wall Activation (Cross-section)")
plt.axis("equal")
plt.tight_layout()
plt.savefig("results/week3/activation_cross_section.png", dpi=300)
plt.show()
