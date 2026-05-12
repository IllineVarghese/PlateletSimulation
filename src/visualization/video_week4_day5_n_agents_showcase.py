import os
import numpy as np
import pyvista as pv


RESULTS_DIR = "results"
OUTPUT_MOVIE = os.path.join(RESULTS_DIR, "week4_day5_n_agents_showcase.mp4")
OUTPUT_IMAGE = os.path.join(RESULTS_DIR, "week4_day5_n_agents_showcase.png")

DT = 0.1
STEPS = 100
N_AGENTS = 20
FRAME_REPEAT = 2

GRN_INPUT_NODES = {
    "collision_impulse": "InCollisionImpulse",
    "chemical_concentration": "InChemicalConcentration",
    "shear_stress": "InShearStress",
}

GRN_OUTPUT_NODES = {
    "stickiness": "OutStickiness",
    "morphology": "OutCellShapeChange",
    "secretion_rate": "OutSecretionRate",
}

CHEMICAL_SOURCE_POSITION = np.array([1.5, 0.0, 0.0], dtype=float)
CHEMICAL_SOURCE_RADIUS = 0.75

VESSEL_RADIUS = 1.0
VESSEL_LENGTH = 10.0
U_MAX = 1.0

AGENT_COLORS = [
    "royalblue", "limegreen", "darkorange", "mediumorchid", "goldenrod",
    "deepskyblue", "tomato", "turquoise", "violet", "olive",
    "dodgerblue", "chartreuse", "coral", "slateblue", "khaki",
    "cyan", "salmon", "springgreen", "plum", "orange",
]


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def compute_radial_distance(position: np.ndarray) -> float:
    return float(np.linalg.norm(position[1:3]))


def compute_poiseuille_velocity_at_position(position: np.ndarray) -> float:
    r = compute_radial_distance(position)
    if r >= VESSEL_RADIUS:
        return 0.0
    u = U_MAX * (1.0 - (r / VESSEL_RADIUS) ** 2)
    return max(0.0, u)


def compute_poiseuille_shear_from_gradient(position: np.ndarray) -> float:
    r = compute_radial_distance(position)
    raw_gradient = (2.0 * U_MAX * r) / (VESSEL_RADIUS ** 2)
    max_gradient = (2.0 * U_MAX) / VESSEL_RADIUS
    normalized_shear = raw_gradient / max_gradient
    return clamp(normalized_shear)


class DummyGRN:
    def __init__(self):
        self.nodes = {}

    def set_node(self, name: str, value: float) -> None:
        self.nodes[name] = float(value)

    def get_node(self, name: str) -> float:
        return float(self.nodes.get(name, 0.0))

    def step(self, dt: float) -> None:
        collision = self.nodes.get(GRN_INPUT_NODES["collision_impulse"], 0.0)
        chemical = self.nodes.get(GRN_INPUT_NODES["chemical_concentration"], 0.0)
        shear = self.nodes.get(GRN_INPUT_NODES["shear_stress"], 0.0)

        stickiness = clamp(0.55 * collision + 0.15 * chemical + 0.30 * shear)
        morphology = clamp(0.10 * collision + 0.30 * chemical + 0.60 * shear)
        secretion_rate = clamp(0.10 * collision + 0.60 * chemical + 0.30 * shear)

        self.nodes[GRN_OUTPUT_NODES["stickiness"]] = stickiness
        self.nodes[GRN_OUTPUT_NODES["morphology"]] = morphology
        self.nodes[GRN_OUTPUT_NODES["secretion_rate"]] = secretion_rate


class AgentSensors:
    def __init__(self):
        self.collision_impulse = 0.0
        self.chemical_concentration = 0.0
        self.shear_stress = 0.0


class AgentOutputs:
    def __init__(self):
        self.stickiness = 0.0
        self.morphology = 0.0
        self.secretion_rate = 0.0


class GRNAgent:
    def __init__(self, name: str, index: int, position: np.ndarray):
        self.name = name
        self.index = index
        self.position = position.astype(float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)

        self.grn = DummyGRN()
        self.sensors = AgentSensors()
        self.outputs = AgentOutputs()

        self.debug_history = {
            "step": [],
            "position_x": [],
            "position_y": [],
            "position_z": [],
            "flow_velocity": [],
            "collision_impulse": [],
            "chemical_concentration": [],
            "shear_stress": [],
            "stickiness": [],
            "morphology": [],
            "secretion_rate": [],
            "speed": [],
        }


def create_agents(n_agents: int):
    agents = []
    y_positions = np.linspace(0.0, 0.8, n_agents)

    for i in range(n_agents):
        name = f"A{i}"
        position = np.array([0.0, y_positions[i], 0.0], dtype=float)
        agents.append(GRNAgent(name=name, index=i, position=position))

    return agents


def reset_sensors(agent: GRNAgent) -> None:
    agent.sensors.collision_impulse = 0.0
    agent.sensors.chemical_concentration = 0.0
    agent.sensors.shear_stress = 0.0


def compute_collision_impulse(agent: GRNAgent, step: int) -> float:
    start_1 = 10 + 2 * agent.index
    end_1 = start_1 + 8

    start_2 = end_1 + 2
    end_2 = start_2 + 6

    amp_1 = max(0.3, 1.0 - 0.02 * agent.index)
    amp_2 = max(0.15, 0.5 - 0.015 * agent.index)

    if start_1 <= step <= end_1:
        return amp_1
    if start_2 <= step <= end_2:
        return amp_2
    return 0.0


def compute_chemical_concentration(agent: GRNAgent, step: int) -> float:
    distance = np.linalg.norm(agent.position - CHEMICAL_SOURCE_POSITION)
    concentration = 1.0 - (distance / CHEMICAL_SOURCE_RADIUS)
    concentration = clamp(concentration)
    concentration *= max(0.4, 1.0 - 0.03 * agent.index)
    return clamp(concentration)


def compute_shear_stress(agent: GRNAgent, step: int) -> float:
    return compute_poiseuille_shear_from_gradient(agent.position)


def write_sensors_to_grn(agent: GRNAgent) -> None:
    agent.grn.set_node(GRN_INPUT_NODES["collision_impulse"], clamp(agent.sensors.collision_impulse))
    agent.grn.set_node(GRN_INPUT_NODES["chemical_concentration"], clamp(agent.sensors.chemical_concentration))
    agent.grn.set_node(GRN_INPUT_NODES["shear_stress"], clamp(agent.sensors.shear_stress))


def step_agent_grn(agent: GRNAgent, dt: float) -> None:
    agent.grn.step(dt)


def read_grn_outputs(agent: GRNAgent) -> None:
    agent.outputs.stickiness = clamp(agent.grn.get_node(GRN_OUTPUT_NODES["stickiness"]))
    agent.outputs.morphology = clamp(agent.grn.get_node(GRN_OUTPUT_NODES["morphology"]))
    agent.outputs.secretion_rate = clamp(agent.grn.get_node(GRN_OUTPUT_NODES["secretion_rate"]))


def apply_agent_outputs(agent: GRNAgent, dt: float) -> None:
    local_flow_velocity = compute_poiseuille_velocity_at_position(agent.position)
    mobility_factor = max(0.15, 1.0 - 0.55 * agent.outputs.stickiness)

    agent.velocity[0] = local_flow_velocity * mobility_factor
    agent.velocity[1] = 0.0
    agent.velocity[2] = 0.0


def update_position(agent: GRNAgent, dt: float) -> None:
    agent.position += agent.velocity * dt
    agent.position[0] = min(agent.position[0], VESSEL_LENGTH)


def record_agent_debug(agent: GRNAgent, step: int) -> None:
    flow_velocity = compute_poiseuille_velocity_at_position(agent.position)

    agent.debug_history["step"].append(step)
    agent.debug_history["position_x"].append(float(agent.position[0]))
    agent.debug_history["position_y"].append(float(agent.position[1]))
    agent.debug_history["position_z"].append(float(agent.position[2]))
    agent.debug_history["flow_velocity"].append(flow_velocity)
    agent.debug_history["collision_impulse"].append(agent.sensors.collision_impulse)
    agent.debug_history["chemical_concentration"].append(agent.sensors.chemical_concentration)
    agent.debug_history["shear_stress"].append(agent.sensors.shear_stress)
    agent.debug_history["stickiness"].append(agent.outputs.stickiness)
    agent.debug_history["morphology"].append(agent.outputs.morphology)
    agent.debug_history["secretion_rate"].append(agent.outputs.secretion_rate)
    agent.debug_history["speed"].append(float(np.linalg.norm(agent.velocity)))


def update_one_agent(agent: GRNAgent, step: int, dt: float) -> None:
    reset_sensors(agent)
    agent.sensors.collision_impulse = compute_collision_impulse(agent, step)
    agent.sensors.chemical_concentration = compute_chemical_concentration(agent, step)
    agent.sensors.shear_stress = compute_shear_stress(agent, step)

    write_sensors_to_grn(agent)
    step_agent_grn(agent, dt)
    read_grn_outputs(agent)

    apply_agent_outputs(agent, dt)
    update_position(agent, dt)
    record_agent_debug(agent, step)


def run_simulation(n_agents: int):
    agents = create_agents(n_agents)
    for step in range(STEPS):
        for agent in agents:
            update_one_agent(agent, step, DT)
    return agents


def make_vessel_surface():
    return pv.Cylinder(
        center=(VESSEL_LENGTH / 2.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=VESSEL_RADIUS,
        height=VESSEL_LENGTH,
        resolution=96,
    )


def make_end_ring(x_pos: float, radius: float, n_points: int = 120):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    points = np.column_stack([
        np.full_like(theta, x_pos),
        radius * np.cos(theta),
        radius * np.sin(theta),
    ])
    poly = pv.PolyData(points)
    poly.lines = np.hstack([[n_points], np.arange(n_points)])
    return poly


def make_flow_arrows():
    xs = np.linspace(1.0, VESSEL_LENGTH - 1.0, 4)
    ys = np.linspace(-VESSEL_RADIUS * 0.65, VESSEL_RADIUS * 0.65, 4)
    zs = np.linspace(-VESSEL_RADIUS * 0.65, VESSEL_RADIUS * 0.65, 4)

    pts = []
    vecs = []

    for x in xs:
        for y in ys:
            for z in zs:
                r = np.sqrt(y**2 + z**2)
                if r > VESSEL_RADIUS:
                    continue
                u = U_MAX * (1.0 - (r / VESSEL_RADIUS) ** 2)
                if u < 0.10:
                    continue
                pts.append([x, y, z])
                vecs.append([u, 0.0, 0.0])

    pts = np.array(pts, dtype=float)
    vecs = np.array(vecs, dtype=float)

    pdata = pv.PolyData(pts)
    pdata["vectors"] = vecs
    pdata["mag"] = np.linalg.norm(vecs, axis=1)
    return pdata.glyph(orient="vectors", scale="mag", factor=0.32)


def make_centerline():
    return pv.Line((0.0, 0.0, 0.0), (VESSEL_LENGTH, 0.0, 0.0), resolution=1)


def make_source_mesh():
    return pv.Sphere(radius=0.05, center=CHEMICAL_SOURCE_POSITION)


def make_chemical_field():
    grid_size = 14

    x = np.linspace(
        CHEMICAL_SOURCE_POSITION[0] - CHEMICAL_SOURCE_RADIUS,
        CHEMICAL_SOURCE_POSITION[0] + CHEMICAL_SOURCE_RADIUS,
        grid_size,
    )
    y = np.linspace(-CHEMICAL_SOURCE_RADIUS, CHEMICAL_SOURCE_RADIUS, grid_size)
    z = np.linspace(-CHEMICAL_SOURCE_RADIUS, CHEMICAL_SOURCE_RADIUS, grid_size)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    distances = np.sqrt(
        (X - CHEMICAL_SOURCE_POSITION[0]) ** 2
        + (Y - CHEMICAL_SOURCE_POSITION[1]) ** 2
        + (Z - CHEMICAL_SOURCE_POSITION[2]) ** 2
    )

    concentration = 1.0 - (distances / CHEMICAL_SOURCE_RADIUS)
    concentration = np.clip(concentration, 0.0, 1.0)

    grid = pv.StructuredGrid(X, Y, Z)
    grid["concentration"] = concentration.flatten(order="F")
    return grid


def make_agent_sphere(center):
    return pv.Sphere(radius=0.10, center=center, theta_resolution=24, phi_resolution=24)


def make_outline_sphere(center):
    return pv.Sphere(radius=0.112, center=center, theta_resolution=24, phi_resolution=24)


def make_trail_mesh(history_x, history_y, history_z):
    points = np.column_stack([history_x, history_y, history_z])
    if len(points) < 2:
        return None
    return pv.Spline(points, len(points) * 4)


def add_static_scene(plotter):
    vessel = make_vessel_surface()
    inlet_ring = make_end_ring(0.0, VESSEL_RADIUS)
    outlet_ring = make_end_ring(VESSEL_LENGTH, VESSEL_RADIUS)
    source = make_source_mesh()
    flow_arrows = make_flow_arrows()
    centerline = make_centerline()
    chemical_field = make_chemical_field()
    chemical_contours = chemical_field.contour(isosurfaces=[0.4, 0.7])

    plotter.add_mesh(vessel, color="lightsteelblue", opacity=0.10, smooth_shading=True)
    plotter.add_mesh(inlet_ring, color="navy", line_width=3)
    plotter.add_mesh(outlet_ring, color="navy", line_width=3)
    plotter.add_mesh(flow_arrows, color="lightsteelblue", opacity=0.14)
    plotter.add_mesh(centerline, color="black", line_width=2)

    plotter.add_mesh(chemical_field, scalars="concentration", cmap="Reds", opacity=0.015, show_scalar_bar=False)
    plotter.add_mesh(chemical_contours, cmap="Reds", opacity=0.04, show_scalar_bar=False)
    plotter.add_mesh(source, color="red", smooth_shading=True)

    plotter.show_axes()
    plotter.show_bounds(
        grid="front",
        location="outer",
        xtitle="X (flow direction)",
        ytitle="Y",
        ztitle="Z",
        all_edges=True,
        font_size=10,
    )
    plotter.add_text("Week 4 Day 5: Larger N-Agent 3D Showcase", position="upper_left", font_size=14)


def render_movie(agents, output_path: str, image_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 832))
    plotter.set_background("white")
    plotter.open_movie(output_path, framerate=15)

    add_static_scene(plotter)

    outline_actors = []
    sphere_actors = []
    trail_actors = [None] * len(agents)

    for i, agent in enumerate(agents):
        start_center = (
            agent.debug_history["position_x"][0],
            agent.debug_history["position_y"][0],
            agent.debug_history["position_z"][0],
        )

        outline_actor = plotter.add_mesh(make_outline_sphere(start_center), color="dimgray", smooth_shading=True)
        sphere_actor = plotter.add_mesh(
            make_agent_sphere(start_center),
            color=AGENT_COLORS[i % len(AGENT_COLORS)],
            smooth_shading=True,
        )

        outline_actors.append(outline_actor)
        sphere_actors.append(sphere_actor)

    plotter.camera_position = [
        (5.0, -6.5, 3.4),
        (5.0, 0.35, 0.0),
        (0.0, 0.0, 1.0),
    ]

    n_steps = len(agents[0].debug_history["step"])
    snapshot_frame = n_steps // 2

    for step_idx in range(n_steps):
        for i, agent in enumerate(agents):
            center = (
                agent.debug_history["position_x"][step_idx],
                agent.debug_history["position_y"][step_idx],
                agent.debug_history["position_z"][step_idx],
            )

            outline_actors[i].mapper.SetInputData(make_outline_sphere(center))
            sphere_actors[i].mapper.SetInputData(make_agent_sphere(center))

            if trail_actors[i] is not None:
                plotter.remove_actor(trail_actors[i])

            trail = make_trail_mesh(
                agent.debug_history["position_x"][: step_idx + 1],
                agent.debug_history["position_y"][: step_idx + 1],
                agent.debug_history["position_z"][: step_idx + 1],
            )
            if trail is not None:
                trail_actors[i] = plotter.add_mesh(
                    trail,
                    color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    line_width=2,
                )

        if step_idx == snapshot_frame:
            plotter.screenshot(image_path)

        for _ in range(FRAME_REPEAT):
            plotter.write_frame()

    plotter.close()


def print_summary(agents) -> None:
    print("\n=== Week 4 Day 5: Larger N-Agent 3D Showcase ===")
    print("Total agents:", len(agents))
    print("Saved movie:", OUTPUT_MOVIE)
    print("Saved image:", OUTPUT_IMAGE)


if __name__ == "__main__":
    agents = run_simulation(N_AGENTS)
    render_movie(agents, OUTPUT_MOVIE, OUTPUT_IMAGE)
    print_summary(agents)