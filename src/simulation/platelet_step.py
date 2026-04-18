import warp as wp
import numpy as np

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent
from src.simulation.chemical_field import ChemicalField


# --------------------------------------------------
# Shear stress approximation (simple radial model)
# Assumption:
# - vessel cross-section is in x-y plane
# - vessel center is around (0.5, 0.5)
# - radius is 0.5 because positions are initialized in [0, 1]
# --------------------------------------------------
def compute_radial_distance(pos, center=(0.5, 0.5)):
    dx = float(pos[0]) - float(center[0])
    dy = float(pos[1]) - float(center[1])
    return np.sqrt(dx * dx + dy * dy)


def compute_shear_stress(pos, center=(0.5, 0.5), radius=0.5):
    r = compute_radial_distance(pos, center)
    shear = r / radius
    return max(0.0, min(1.0, float(shear)))


# --------------------------------------------------
# Warp kernel: movement slowed by adhesion strength
# --------------------------------------------------
@wp.kernel
def move_platelets(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    adhesion_strengths: wp.array(dtype=float),
    dt: float,
):
    i = wp.tid()

    slowdown = 1.0 - adhesion_strengths[i]

    if slowdown < 0.0:
        slowdown = 0.0
    if slowdown > 1.0:
        slowdown = 1.0

    positions[i] = positions[i] + velocities[i] * slowdown * dt


# --------------------------------------------------
# Global simulation state
# Keeps positions, velocities, and GRN agents alive
# across multiple run_step() calls
# --------------------------------------------------
_SIM_STATE = None


def _initialize_sim(device: str = "cpu"):
    global _SIM_STATE

    wp.init()

    if device == "cuda" and wp.is_cuda_available():
        wp.set_device("cuda")
    else:
        wp.set_device("cpu")

    num_platelets = 80
    dt = 0.01

    rng = np.random.default_rng(42)

    positions_np = rng.random((num_platelets, 3), dtype=np.float32)
    velocities_np = (rng.normal(size=(num_platelets, 3)) * 0.1).astype(np.float32)

    positions = wp.array(
        positions_np,
        dtype=wp.vec3,
        device=wp.get_device(),
    )

    velocities = wp.array(
        velocities_np,
        dtype=wp.vec3,
        device=wp.get_device(),
    )

    adhesion_np = np.zeros(num_platelets, dtype=np.float32)
    adhesion_strengths = wp.array(
        adhesion_np,
        dtype=float,
        device=wp.get_device(),
    )

    model = load_graphml("data/networks/test_minimal.graphml")
    agents = [GRNAgent(model) for _ in range(num_platelets)]

    _SIM_STATE = {
        "positions": positions,
        "velocities": velocities,
        "adhesion_strengths": adhesion_strengths,
        "agents": agents,
        "num_platelets": num_platelets,
        "dt": dt,
        "step": 0,
        "device": device,
        "chemical_field": ChemicalField(),
    }


def _get_collision_input(step: int, platelet_index: int) -> float:
    """
    Simple prototype collision pattern.

    Platelets receive different impulses over time so GRN can
    produce different stickiness values.
    """
    if (step + platelet_index) % 5 == 0:
        return 1.0
    elif (step + platelet_index) % 3 == 0:
        return 0.5
    else:
        return 0.0


def run_step(device: str = "cpu"):
    global _SIM_STATE

    if _SIM_STATE is None:
        _initialize_sim(device)

    positions = _SIM_STATE["positions"]
    velocities = _SIM_STATE["velocities"]
    agents = _SIM_STATE["agents"]
    num_platelets = _SIM_STATE["num_platelets"]
    dt = _SIM_STATE["dt"]
    step = _SIM_STATE["step"]
    chemical_field = _SIM_STATE["chemical_field"]

    positions_np = positions.numpy()

    chemical_field.decay(dt)

    adhesion_np = np.zeros(num_platelets, dtype=np.float32)
    secretion_np = np.zeros(num_platelets, dtype=np.float32)
    chemical_input_np = np.zeros(num_platelets, dtype=np.float32)

    for i, agent in enumerate(agents):
        current_pos = positions_np[i]

        collision_input = 0.0
        shear_input = compute_shear_stress(current_pos)
        molecule_input = chemical_field.sample(current_pos)

        chemical_input_np[i] = molecule_input

        agent.set_sensor("InCollisionImpulse", collision_input)
        agent.set_sensor("InShearStress", shear_input)
        agent.set_sensor("InMolecule", molecule_input)

        for _ in range(10):
            agent.step()

        stickiness = agent.get_output("OutStickiness")
        secretion_rate = agent.get_output("OutSecretionRate")

        adhesion_strength = min(1.0, max(0.0, float(stickiness)))
        secretion_amount = max(0.0, float(secretion_rate)) * dt

        adhesion_np[i] = adhesion_strength
        secretion_np[i] = secretion_amount

    if i < 10:
        print(
            f"Platelet {i}: "
            f"shear={shear_input:.3f}, "
            f"chemical={molecule_input:.4f}, "
            f"stickiness={stickiness:.6f}, "
            f"secretion={secretion_rate:.6f}"
        )   

    for i in range(num_platelets):
        chemical_field.deposit(positions_np[i], secretion_np[i])

    adhesion_strengths = wp.array(
        adhesion_np,
        dtype=float,
        device=wp.get_device(),
    )

    _SIM_STATE["adhesion_strengths"] = adhesion_strengths

    wp.launch(
        kernel=move_platelets,
        dim=num_platelets,
        inputs=[positions, velocities, adhesion_strengths, dt],
    )

    _SIM_STATE["step"] += 1

    print(f"Simulation step: {_SIM_STATE['step']}")
    print("First 10 adhesion strengths:")
    print(adhesion_np[:10])
    print("First 10 secretion amounts:")
    print(secretion_np[:10])
    print("First 10 sampled chemical inputs:")
    print(chemical_input_np[:10])
    print("Updated first 5 positions:")
    print(positions.numpy()[:5])

    return positions, adhesion_strengths

if __name__ == "__main__":
    run_step("cpu")