import warp as wp
import numpy as np

from src.grn_engine.graphml_parser import load_graphml
from src.grn_engine.agent_grn import GRNAgent


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

    num_platelets = 10
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

    # ----------------------------------------------
    # GRN update: OutStickiness -> adhesion strength
    # ----------------------------------------------
    adhesion_np = np.zeros(num_platelets, dtype=np.float32)

    for i, agent in enumerate(agents):
        collision_input = _get_collision_input(step, i)

        agent.set_sensor("InCollisionImpulse", collision_input)
        agent.step()

        stickiness = agent.get_output("OutStickiness")

        current_pos = positions.numpy()[i]
        y_pos = current_pos[1]

        # wider wall region
        near_wall = (y_pos < 0.4) or (y_pos > 0.6)

        if near_wall:
           adhesion_strength = min(1.0, float(stickiness) * 3.0)
        else:
           adhesion_strength = min(1.0, float(stickiness))

        adhesion_strength = max(0.0, adhesion_strength)
        adhesion_np[i] = adhesion_strength
        
    print(f"Platelet {i}: y={y_pos:.3f}, near_wall={near_wall}, stickiness={stickiness:.4f}, adhesion={adhesion_strength:.4f}")

    adhesion_strengths = wp.array(
        adhesion_np,
        dtype=float,
        device=wp.get_device(),
    )

    _SIM_STATE["adhesion_strengths"] = adhesion_strengths

    # ----------------------------------------------
    # Movement update slowed by adhesion strength
    # ----------------------------------------------
    wp.launch(
        kernel=move_platelets,
        dim=num_platelets,
        inputs=[positions, velocities, adhesion_strengths, dt],
    )

    _SIM_STATE["step"] += 1

    print(f"Simulation step: {_SIM_STATE['step']}")
    print("Adhesion strengths:")
    print(adhesion_np)
    print("Updated positions:")
    print(positions.numpy())

    # return both so platelet_sim.py can save both
    return positions, adhesion_strengths


if __name__ == "__main__":
    run_step("cpu")