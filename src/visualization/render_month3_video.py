import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = Path("results/month3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "month3_behavior_shear_analysis_3d.mp4"
OUTPUT_SNAPSHOT = OUTPUT_DIR / "month3_behavior_shear_analysis_3d_snapshot.png"


# ============================================================
# Simulation settings
# ============================================================

N_AGENTS = 24
N_STEPS = 520
DT = 0.08
FPS = 20

VESSEL_LENGTH = 10.0
VESSEL_RADIUS = 1.0

BASE_FLOW_SPEED = 0.045
FLOW_SCALE = 1.8

TRAIL_LENGTH = 35
CAMERA_ROTATION = True

np.random.seed(7)


# ============================================================
# Initial agent state
# ============================================================

x = np.random.uniform(0.0, 2.0, N_AGENTS)

theta = np.random.uniform(0.0, 2.0 * np.pi, N_AGENTS)
r = np.random.uniform(0.1, 0.9, N_AGENTS)

y = r * np.cos(theta)
z = r * np.sin(theta)

phase = np.random.uniform(0.0, 2.0 * np.pi, N_AGENTS)
agent_bias = np.random.uniform(0.85, 1.15, N_AGENTS)


history_x = []
history_y = []
history_z = []
history_stickiness = []
history_secretion = []
history_morphology = []


# ============================================================
# Helper functions
# ============================================================

def sigmoid(v, gain=2.2):
    return 1.0 / (1.0 + np.exp(-gain * v))


def radial_distance(y_pos, z_pos):
    return np.sqrt(y_pos ** 2 + z_pos ** 2)


def shear_from_radius(y_pos, z_pos):
    return np.clip(radial_distance(y_pos, z_pos) / VESSEL_RADIUS, 0.0, 1.0)


def chemical_field(x_pos, y_pos, z_pos, step):
    source_x = 1.5 + 7.0 * (step / N_STEPS)
    source_y = 0.30 * math.sin(step * 0.025)
    source_z = 0.30 * math.cos(step * 0.025)

    dx = x_pos - source_x
    dy = y_pos - source_y
    dz = z_pos - source_z

    concentration = np.exp(-(dx * dx + dy * dy + dz * dz) / 1.35)
    return np.clip(concentration, 0.0, 1.0)


def poiseuille_velocity(y_pos, z_pos):
    rr = np.clip(radial_distance(y_pos, z_pos), 0.0, 1.0)
    return BASE_FLOW_SPEED + FLOW_SCALE * (0.035 + 0.080 * (1.0 - rr * rr))


# ============================================================
# Simulate
# ============================================================

for step in range(N_STEPS):
    shear = shear_from_radius(y, z)
    chemical = chemical_field(x, y, z, step)

    activation_drive = (
        0.70 * shear
        + 0.55 * chemical
        + 0.18 * np.sin(step * 0.030 + phase)
    ) * agent_bias

    activation = sigmoid(activation_drive - 0.50, gain=2.2)

    stickiness = 0.32 + 0.43 * activation - 0.20 * shear
    secretion = 0.22 + 0.55 * activation + 0.22 * chemical
    morphology = 0.28 + 0.62 * activation

    stickiness = np.clip(stickiness, 0.0, 1.0)
    secretion = np.clip(secretion, 0.0, 1.0)
    morphology = np.clip(morphology, 0.0, 1.0)

    vx = poiseuille_velocity(y, z) * (1.0 - 0.28 * stickiness)

    angular_motion = 0.22 * np.sin(step * 0.030 + phase)
    radial_pull = -0.010 * shear

    y_drift = (
        0.018 * np.sin(step * 0.035 + phase)
        + radial_pull * y
        + 0.006 * angular_motion * z
    )

    z_drift = (
        0.018 * np.cos(step * 0.035 + phase)
        + radial_pull * z
        - 0.006 * angular_motion * y
    )

    x = x + vx * DT
    y = y + y_drift * DT
    z = z + z_drift * DT

    rr = radial_distance(y, z)
    outside = rr > 0.95
    if np.any(outside):
        y[outside] = y[outside] / rr[outside] * 0.95
        z[outside] = z[outside] / rr[outside] * 0.95

    wrapped = x > VESSEL_LENGTH
    if np.any(wrapped):
        x[wrapped] = 0.0
        new_theta = np.random.uniform(0.0, 2.0 * np.pi, wrapped.sum())
        new_r = np.random.uniform(0.1, 0.9, wrapped.sum())
        y[wrapped] = new_r * np.cos(new_theta)
        z[wrapped] = new_r * np.sin(new_theta)
        phase[wrapped] = np.random.uniform(0.0, 2.0 * np.pi, wrapped.sum())

    history_x.append(x.copy())
    history_y.append(y.copy())
    history_z.append(z.copy())
    history_stickiness.append(stickiness.copy())
    history_secretion.append(secretion.copy())
    history_morphology.append(morphology.copy())


history_x = np.array(history_x)
history_y = np.array(history_y)
history_z = np.array(history_z)
history_stickiness = np.array(history_stickiness)
history_secretion = np.array(history_secretion)
history_morphology = np.array(history_morphology)


# ============================================================
# 3D rendering
# ============================================================

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111, projection="3d")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_xlim(0.0, VESSEL_LENGTH)
ax.set_ylim(-1.15, 1.15)
ax.set_zlim(-1.15, 1.15)

ax.set_xlabel("Axial vessel position")
ax.set_ylabel("Y radial position")
ax.set_zlabel("Z radial position")
ax.set_title("Month 3: 3D GRN-driven platelet behavior under shear flow")

# vessel cylinder wireframe
x_cyl = np.linspace(0, VESSEL_LENGTH, 50)
theta_cyl = np.linspace(0, 2.0 * np.pi, 55)
Xc, Tc = np.meshgrid(x_cyl, theta_cyl)
Yc = VESSEL_RADIUS * np.cos(Tc)
Zc = VESSEL_RADIUS * np.sin(Tc)

ax.plot_wireframe(Xc, Yc, Zc, color="gray", alpha=0.14, linewidth=0.45)

# centerline
ax.plot(
    [0, VESSEL_LENGTH],
    [0, 0],
    [0, 0],
    linestyle="--",
    color="black",
    alpha=0.35,
    linewidth=1,
)

sizes = 85 + 280 * history_morphology[0]

scatter = ax.scatter(
    history_x[0],
    history_y[0],
    history_z[0],
    s=sizes,
    c=history_secretion[0],
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    edgecolors="black",
    linewidths=0.6,
    depthshade=True,
)

cbar = plt.colorbar(scatter, ax=ax, pad=0.08, shrink=0.72)
cbar.set_label("Secretion output")

text_box = ax.text2D(
    0.03,
    0.94,
    "",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
)

legend_text = ax.text2D(
    0.03,
    0.06,
    "Dot size = morphology output\n"
    "Dot color = secretion output\n"
    "Wall proximity = shear exposure\n"
    "3D vessel = cylindrical cross-section",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
)

ax.view_init(elev=23, azim=-65)


def draw_static_scene():
    ax.plot_wireframe(Xc, Yc, Zc, color="gray", alpha=0.14, linewidth=0.45)
    ax.plot(
        [0, VESSEL_LENGTH],
        [0, 0],
        [0, 0],
        linestyle="--",
        color="black",
        alpha=0.35,
        linewidth=1,
    )


def update(frame):
    ax.cla()

    ax.set_xlim(0.0, VESSEL_LENGTH)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)

    ax.set_xlabel("Axial vessel position")
    ax.set_ylabel("Y radial position")
    ax.set_zlabel("Z radial position")
    ax.set_title("Month 3: 3D GRN-driven platelet behavior under shear flow")

    draw_static_scene()

    start = max(0, frame - TRAIL_LENGTH)

    for i in range(N_AGENTS):
        ax.plot(
            history_x[start:frame + 1, i],
            history_y[start:frame + 1, i],
            history_z[start:frame + 1, i],
            color="black",
            alpha=0.18,
            linewidth=0.8,
        )

    sizes = 85 + 280 * history_morphology[frame]

    sc = ax.scatter(
        history_x[frame],
        history_y[frame],
        history_z[frame],
        s=sizes,
        c=history_secretion[frame],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        linewidths=0.6,
        depthshade=True,
    )

    mean_shear = np.mean(shear_from_radius(history_y[frame], history_z[frame]))
    mean_stickiness = np.mean(history_stickiness[frame])
    mean_secretion = np.mean(history_secretion[frame])
    mean_morphology = np.mean(history_morphology[frame])

    ax.text2D(
        0.03,
        0.94,
        f"step = {frame}\n"
        f"mean shear = {mean_shear:.3f}\n"
        f"mean stickiness = {mean_stickiness:.3f}\n"
        f"mean secretion = {mean_secretion:.3f}\n"
        f"mean morphology = {mean_morphology:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    ax.text2D(
        0.03,
        0.06,
        "Dot size = morphology output\n"
        "Dot color = secretion output\n"
        "Wall proximity = shear exposure\n"
        "Trails = platelet transport history",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    if CAMERA_ROTATION:
        ax.view_init(elev=23, azim=-65 + frame * 0.07)
    else:
        ax.view_init(elev=23, azim=-65)

    return sc,


animation = FuncAnimation(
    fig,
    update,
    frames=N_STEPS,
    interval=1000 / FPS,
    blit=False,
)

writer = FFMpegWriter(
    fps=FPS,
    metadata={
        "title": "Month 3 3D GRN-driven platelet behavior under shear flow",
        "artist": "PlateletSimulation",
    },
    bitrate=4500,
)

animation.save(OUTPUT_VIDEO, writer=writer)

update(N_STEPS - 1)
plt.savefig(OUTPUT_SNAPSHOT, dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"Saved video: {OUTPUT_VIDEO}")
print(f"Saved snapshot: {OUTPUT_SNAPSHOT}")