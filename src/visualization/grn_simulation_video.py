import matplotlib.pyplot as plt
import numpy as np
import imageio

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


# -----------------------------
# Configuration
# -----------------------------
N_STEPS = 12
FPS = 2
OUTPUT_PATH = "results/week3_grn_simulation.mp4"

# Fixed y-positions for 3 platelets
Y_POS = [0.25, 0.0, -0.25]

# Starting x-positions
X_START = [-0.8, -0.4, 0.0]

# Small flow velocity per platelet (left -> right)
X_VEL = [0.18, 0.16, 0.20]

# Collision pattern over time for each platelet
# 0 = no collision, 1 = strong collision, 0.5 = medium collision
collision_inputs = [
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],      # Platelet 1
    [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0],  # Platelet 2
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],      # Platelet 3
]


# -----------------------------
# Load GRN model and agents
# -----------------------------
model = load_graphml("data/networks/test_minimal.graphml")

agents = [
    GRNAgent(model),
    GRNAgent(model),
    GRNAgent(model),
]


# -----------------------------
# Build frames
# -----------------------------
frames = []

for t in range(N_STEPS):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Channel walls
    ax.axhline(0.55, color="black", linewidth=1.2)
    ax.axhline(-0.55, color="black", linewidth=1.2)

    # Platelet positions move with time
    x_positions = [X_START[i] + X_VEL[i] * t for i in range(3)]
    y_positions = Y_POS

    stickiness_values = []

    # Update each platelet agent
    for i, agent in enumerate(agents):
        agent.set_sensor("InCollisionImpulse", collision_inputs[i][t])
        agent.step()
        stickiness_values.append(agent.get_output("OutStickiness"))

    # Scatter plot
    scatter = ax.scatter(
        x_positions,
        y_positions,
        c=stickiness_values,
        s=900,
        cmap="Reds",
        vmin=0.0,
        vmax=0.2,
        edgecolors="darkred",
        linewidths=1.5,
        zorder=3,
    )

    # Platelet labels and values
    for i, val in enumerate(stickiness_values):
        ax.text(
            x_positions[i],
            y_positions[i] + 0.14,
            f"P{i+1}\nStick={val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    # Collision input labels
    for i in range(3):
        ax.text(
            x_positions[i],
            y_positions[i] - 0.16,
            f"Input={collision_inputs[i][t]}",
            ha="center",
            va="top",
            fontsize=9,
        )

    # Add arrows for active collisions
    for i in range(3):
        if collision_inputs[i][t] > 0:
            ax.annotate(
                "",
                xy=(x_positions[i], y_positions[i] + 0.06),
                xytext=(x_positions[i], y_positions[i] + 0.22),
                arrowprops=dict(arrowstyle="->", color="blue", linewidth=2),
            )

    # Flow direction arrow
    ax.annotate(
        "Flow",
        xy=(2.2, 0.48),
        xytext=(1.4, 0.48),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        fontsize=11,
        ha="center",
    )

    # Title and status text
    ax.set_title(f"Month 2 GRN Platelet Simulation  |  Step {t}", fontsize=15, pad=12)

    active = []
    for i in range(3):
        if collision_inputs[i][t] > 0:
            active.append(f"P{i+1}")

    active_text = ", ".join(active) if active else "None"

    ax.text(
        -1.0,
        0.68,
        f"Active collision inputs: {active_text}",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Clean axes
    ax.set_xlim(-1.2, 2.6)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xticks([])
    ax.set_yticks([])

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("OutStickiness", fontsize=11)

    # Convert figure to image
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    frames.append(image)

    plt.close(fig)


# -----------------------------
# Save video
# -----------------------------
imageio.mimsave(OUTPUT_PATH, frames, fps=FPS)

print(f"Saved video to: {OUTPUT_PATH}")