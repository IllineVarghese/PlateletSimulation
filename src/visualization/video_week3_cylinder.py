import numpy as np
import pyvista as pv

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


# -----------------------------
# Configuration
# -----------------------------
N_STEPS = 12
FPS = 2
FRAME_REPEAT = 3
OUTPUT_PATH = "results/week3_cylinder_grn.mp4"

# Vessel geometry
CYL_RADIUS = 0.55

# Platelet positions in simulation coordinates
# x = vessel axis (flow direction)
# y, z = cross-section positions
X_START = np.array([-1.00, -0.70, -0.40], dtype=float)

YZ_POS = np.array([
    [0.18,  0.10],   # P1
    [0.00, -0.10],   # P2
    [-0.20, 0.15],   # P3
], dtype=float)

# Slower Poiseuille-style flow for presentation clarity
U_MAX = 0.08

# More clearly separated collision patterns
collision_inputs = [
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],        # P1: early strong activation
    [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0],  # P2: later medium activation
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],        # P3: repeated pulsed activation
]


# -----------------------------
# Load GRN model and agents
# -----------------------------
model = load_graphml("data/networks/test_minimal.graphml")
agents = [GRNAgent(model), GRNAgent(model), GRNAgent(model)]


# -----------------------------
# Poiseuille-style speed
# -----------------------------
def poiseuille_speed(y, z, radius, u_max):
    r = np.sqrt(y**2 + z**2)
    r_norm = min(r / radius, 1.0)
    return u_max * (1.0 - r_norm**2)


# -----------------------------
# Precompute trajectories + GRN outputs
# -----------------------------
positions = []
stickiness_history = []
active_inputs = []

current_x = X_START.copy()

for t in range(N_STEPS):
    step_positions = []
    step_stickiness = []
    active = []

    for i, agent in enumerate(agents):
        inp = collision_inputs[i][t]
        agent.set_sensor("InCollisionImpulse", inp)
        agent.step()

        stick = float(agent.get_output("OutStickiness"))
        step_stickiness.append(stick)

        if inp > 0:
            active.append(f"P{i+1}")

        y, z = YZ_POS[i]
        speed = poiseuille_speed(y, z, CYL_RADIUS, U_MAX)
        current_x[i] += speed

        # simulation coordinates: (x, y, z)
        step_positions.append([current_x[i], y, z])

    positions.append(step_positions)
    stickiness_history.append(step_stickiness)
    active_inputs.append(", ".join(active) if active else "None")

positions = np.array(positions, dtype=float)                 # (steps, 3, 3)
stickiness_history = np.array(stickiness_history, dtype=float)  # (steps, 3)


# -----------------------------
# Remap for vertical vessel visualization
# simulation x -> visualization z
# simulation y -> visualization x
# simulation z -> visualization y
# -----------------------------
viz_positions = np.zeros_like(positions)
viz_positions[:, :, 0] = positions[:, :, 1]   # x_vis
viz_positions[:, :, 1] = positions[:, :, 2]   # y_vis
viz_positions[:, :, 2] = positions[:, :, 0]   # z_vis

all_pts = viz_positions.reshape(-1, 3)

z_min = float(all_pts[:, 2].min()) - 0.25
z_max = float(all_pts[:, 2].max()) + 0.25
z_center = 0.5 * (z_min + z_max)

x_center = 0.0
y_center = 0.0

cylinder = pv.Cylinder(
    center=(x_center, y_center, z_center),
    direction=(0, 0, 1),
    radius=CYL_RADIUS,
    height=(z_max - z_min),
    resolution=160,
    capping=False,
)

plotter = pv.Plotter(window_size=(1600, 960), off_screen=True)
plotter.set_background("white")
plotter.open_movie(OUTPUT_PATH, framerate=FPS)

plotter.camera_position = [
    (1.35, -2.35, z_center + 1.0),
    (0.0, 0.0, z_center),
    (0, 0, 1),
]

plotter.enable_lightkit()

max_stick = max(0.25, float(stickiness_history.max()) + 1e-6)


# -----------------------------
# Write frames
# -----------------------------
for t in range(N_STEPS):
    pts = viz_positions[t]
    stick = stickiness_history[t]

    cloud = pv.PolyData(pts)
    cloud["stickiness"] = stick

    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    near_wall_count = int(np.sum(r > 0.7 * CYL_RADIUS))

    mean_stick = float(np.mean(stick))
    max_step_stick = float(np.max(stick))

    plotter.clear()

    # Vessel wall
    plotter.add_mesh(
        cylinder,
        color="lightgray",
        opacity=0.10,
        smooth_shading=True,
    )
    plotter.add_mesh(
        cylinder,
        style="wireframe",
        color="black",
        line_width=0.5,
        opacity=0.18,
    )

    # Platelets
    plotter.add_points(
        cloud,
        scalars="stickiness",
        cmap="Reds",
        render_points_as_spheres=True,
        point_size=26,
        clim=[0.0, max_stick],
        scalar_bar_args={
            "title": "GRN Stickiness",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.18,
            "height": 0.50,
            "width": 0.05,
            "title_font_size": 16,
            "label_font_size": 12,
            "color": "black",
        },
    )

    # Blue arrows for active collision inputs
    for i in range(3):
        if collision_inputs[i][t] > 0:
            px, py, pz = pts[i]
            arrow = pv.Arrow(
                start=(px, py, pz + 0.18),
                direction=(0, 0, -1),
                scale=0.10,
            )
            plotter.add_mesh(arrow, color="blue")

    # Flow arrow
    flow_arrow = pv.Arrow(
        start=(-0.90, -0.82, z_min + 0.20),
        direction=(0, 0, 1),
        scale=0.35,
    )
    plotter.add_mesh(flow_arrow, color="black")

    # Cleaner labels with small offsets so they don't overlap
    label_offsets = [
        np.array([0.03, 0.03, 0.08]),
        np.array([0.03, -0.04, 0.08]),
        np.array([-0.04, 0.03, 0.08]),
    ]

    label_points = []
    label_texts = []

    for i in range(3):
        lp = pts[i] + label_offsets[i]
        label_points.append(lp)
        label_texts.append(
            f"P{i+1}\nStick={stick[i]:.2f}\nInput={collision_inputs[i][t]}"
        )

    plotter.add_point_labels(
        np.array(label_points),
        label_texts,
        font_size=12,
        text_color="black",
        point_color="white",
        point_size=1,
        shape=None,
        always_visible=True,
    )

    # Title + summary text
    plotter.add_text(
        "Week 3: GRN Activation in 3D Cylindrical Poiseuille-Style Flow",
        position=(180, 915),
        font_size=20,
        color="black",
    )

    plotter.add_text(
        "Collision input -> GraphML GRN -> OutStickiness",
        position=(180, 885),
        font_size=13,
        color="black",
    )

    plotter.add_text(
        f"Active collision inputs: {active_inputs[t]}",
        position=(40, 845),
        font_size=8,
        color="black",
    )

    plotter.add_text(
        f"P1 stickiness: {stick[0]:.5f}\n"
        f"P2 stickiness: {stick[1]:.5f}\n"
        f"P3 stickiness: {stick[2]:.5f}",
        position=(40, 800),
        font_size=8,
        color="black",
    )

    plotter.add_text(
        f"Mean stickiness: {mean_stick:.5f}\n"
        f"Max stickiness:  {max_step_stick:.5f}\n"
        f"Near-wall platelets: {near_wall_count}",
        position=(40, 735),
        font_size=8,
        color="black",
    )

    plotter.add_text(
        "Darker red = higher GRN activation / stickiness",
        position=(40, 690),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        f"Step {t+1}/{N_STEPS}",
        position=(40, 45),
        font_size=13,
        color="black",
    )

    plotter.add_text(
        "Flow direction",
        position=(70, 110),
        font_size=11,
        color="black",
    )

    for _ in range(FRAME_REPEAT):
        plotter.write_frame()

plotter.close()

print(f"Saved video to: {OUTPUT_PATH}")