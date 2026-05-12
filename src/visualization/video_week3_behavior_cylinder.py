import numpy as np
import pyvista as pv

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent
from src.simulation.chemical_field import ChemicalField


# -----------------------------
# Configuration
# -----------------------------
N_STEPS = 20
FPS = 3
FRAME_REPEAT = 2
OUTPUT_PATH = "results/week3_behavior_cylinder_clean.mp4"

# Vessel geometry
CYL_RADIUS = 0.55

# Platelet start positions in simulation coordinates
# simulation convention:
# x = vessel axis (flow direction)
# y, z = cross-section
X_START = np.array([-1.00, -0.70, -0.40], dtype=float)

YZ_POS = np.array([
    [0.18,  0.10],   # P1
    [0.00, -0.10],   # P2
    [-0.20, 0.15],   # P3
], dtype=float)

U_MAX = 0.08
SECRETION_SCALE = 1.0
CHEMICAL_DECAY_DT = 0.01

# Optional collision pulses to enrich the demo
collision_inputs = [
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


# -----------------------------
# Helpers
# -----------------------------
def poiseuille_speed(y, z, radius, u_max):
    r = np.sqrt(y**2 + z**2)
    r_norm = min(r / radius, 1.0)
    return u_max * (1.0 - r_norm**2)


def compute_shear_stress_from_yz(y, z, radius):
    r = np.sqrt(y**2 + z**2)
    shear = r / radius
    return max(0.0, min(1.0, float(shear)))


# -----------------------------
# Load GRN model and agents
# -----------------------------
model = load_graphml("data/networks/test_minimal.graphml")
agents = [GRNAgent(model), GRNAgent(model), GRNAgent(model)]
chemical_field = ChemicalField(nx=28, ny=28, nz=60, spacing=0.05, decay_rate=0.1)


# -----------------------------
# Precompute trajectories + GRN outputs
# -----------------------------
positions = []
stickiness_history = []
secretion_history = []
chemical_input_history = []
shear_history = []
active_inputs = []

current_x = X_START.copy()

for t in range(N_STEPS):
    step_positions = []
    step_stickiness = []
    step_secretion = []
    step_chemical = []
    step_shear = []
    active = []

    chemical_field.decay(CHEMICAL_DECAY_DT)

    for i, agent in enumerate(agents):
        y, z = YZ_POS[i]
        x = current_x[i]

        current_pos = np.array([x, y, z], dtype=float)

        collision_input = collision_inputs[i][t]
        shear_input = compute_shear_stress_from_yz(y, z, CYL_RADIUS)
        molecule_input = chemical_field.sample(current_pos)

        agent.set_sensor("InCollisionImpulse", collision_input)
        agent.set_sensor("InShearStress", shear_input)
        agent.set_sensor("InMolecule", molecule_input)

        # multiple GRN substeps so the response is visually clearer
        for _ in range(10):
            agent.step()

        stick = float(agent.get_output("OutStickiness"))
        secretion = float(agent.get_output("OutSecretionRate"))

        secretion_amount = max(0.0, secretion) * CHEMICAL_DECAY_DT * SECRETION_SCALE
        chemical_field.deposit(current_pos, secretion_amount)

        if collision_input > 0:
            active.append(f"P{i+1}")

        speed = poiseuille_speed(y, z, CYL_RADIUS, U_MAX)
        current_x[i] += speed

        step_positions.append([current_x[i], y, z])
        step_stickiness.append(stick)
        step_secretion.append(secretion)
        step_chemical.append(molecule_input)
        step_shear.append(shear_input)

    positions.append(step_positions)
    stickiness_history.append(step_stickiness)
    secretion_history.append(step_secretion)
    chemical_input_history.append(step_chemical)
    shear_history.append(step_shear)
    active_inputs.append(", ".join(active) if active else "None")

positions = np.array(positions, dtype=float)
stickiness_history = np.array(stickiness_history, dtype=float)
secretion_history = np.array(secretion_history, dtype=float)
chemical_input_history = np.array(chemical_input_history, dtype=float)
shear_history = np.array(shear_history, dtype=float)


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

cylinder = pv.Cylinder(
    center=(0.0, 0.0, z_center),
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
    (1.20, -2.10, z_center + 0.90),
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
    secr = secretion_history[t]
    chem = chemical_input_history[t]
    shear = shear_history[t]

    cloud = pv.PolyData(pts)
    cloud["stickiness"] = stick

    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    near_wall_count = int(np.sum(r > 0.7 * CYL_RADIUS))

    mean_stick = float(np.mean(stick))
    mean_secr = float(np.mean(secr))
    mean_chem = float(np.mean(chem))

    plotter.clear()

    # Vessel wall: lighter and less noisy
    plotter.add_mesh(
        cylinder,
        color="lightgray",
        opacity=0.06,
        smooth_shading=True,
    )
    plotter.add_mesh(
        cylinder,
        style="wireframe",
        color="black",
        line_width=0.3,
        opacity=0.08,
    )

    # Platelets colored by adhesion/stickiness
    plotter.add_points(
        cloud,
        scalars="stickiness",
        cmap="Reds",
        render_points_as_spheres=True,
        point_size=34,
        clim=[0.15, max_stick],
        scalar_bar_args={
            "title": "Adhesion / Stickiness",
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
                scale=0.09,
            )
            plotter.add_mesh(arrow, color="blue")

    # Flow arrow
    flow_arrow = pv.Arrow(
        start=(-0.92, -0.82, z_min + 0.18),
        direction=(0, 0, 1),
        scale=0.32,
    )
    plotter.add_mesh(flow_arrow, color="black")

    # Cleaner per-platelet labels
    label_offsets = [
        np.array([0.045, 0.045, 0.10]),
        np.array([0.045, -0.050, 0.10]),
        np.array([-0.055, 0.040, 0.10]),
    ]

    label_points = []
    label_texts = []

    for i in range(3):
        lp = pts[i] + label_offsets[i]
        label_points.append(lp)
        label_texts.append(
            f"P{i+1}\n"
            f"Adh={stick[i]:.3f}\n"
            f"Shear={shear[i]:.2f}\n"
            f"Chem={chem[i]:.4f}"
        )

    plotter.add_point_labels(
        np.array(label_points),
        label_texts,
        font_size=13,
        text_color="black",
        point_color="white",
        point_size=1,
        shape=None,
        always_visible=True,
    )

    # Title and summary
    plotter.add_text(
        "Week 3: 3D Cylindrical Behavior Loop Visualization",
        position=(140, 900),
        font_size=20,
        color="black",
    )

    plotter.add_text(
        "Sensors: collision + shear + chemical  ->  GRN  ->  adhesion + secretion",
        position=(140, 870),
        font_size=13,
        color="black",
    )

    plotter.add_text(
        f"Active collision inputs: {active_inputs[t]}",
        position=(35, 820),
        font_size=10,
        color="black",
    )

    plotter.add_text(
        f"Mean adhesion: {mean_stick:.4f}\n"
        f"Mean secretion: {mean_secr:.4f}\n"
        f"Mean chemical input: {mean_chem:.5f}\n"
        f"Near-wall platelets: {near_wall_count}",
        position=(35, 755),
        font_size=11,
        color="black",
    )

    plotter.add_text(
        "Darker red = higher GRN-controlled adhesion",
        position=(35, 700),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        f"Step {t+1}/{N_STEPS}",
        position=(35, 38),
        font_size=13,
        color="black",
    )

    plotter.add_text(
        "Flow direction",
        position=(65, 100),
        font_size=11,
        color="black",
    )

    for _ in range(FRAME_REPEAT):
        plotter.write_frame()

plotter.close()

print(f"Saved video to: {OUTPUT_PATH}")