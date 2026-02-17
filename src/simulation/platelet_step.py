# src/simulation/platelet_step.py

from __future__ import annotations

from typing import Any, Dict, Optional

import warp as wp

# Optional: keep a module-level init guard (fine)
_WARP_INITIALIZED = False


@wp.kernel
def move_platelets(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt


def step_state(
    state,
    dt: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
    debug_print: bool = False,
) -> None:
    """
    Week 2: Update persistent arrays in-place.

    IMPORTANT:
    - No creation of positions/velocities here.
    - We only launch kernels that update the arrays already stored in `state`.
    """

    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True

    # Decide dt:
    # - If dt is explicitly passed, use it
    # - else if cfg provided, read from cfg
    # - else fall back to a safe default
    if dt is None:
        if cfg is not None:
            dt = cfg.get("simulation", {}).get("dt", cfg.get("dt", 0.01))
        else:
            dt = 0.01

    # Launch on the same device where the state arrays live
    wp.launch(
        kernel=move_platelets,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, float(dt)],
        device=state.device,
    )

    state.step_index += 1

    # Debug print only when requested (WARNING: .numpy() syncs device -> host)
    if debug_print:
        pos_np = state.positions.numpy()
        print(f"[step_state] step={state.step_index} positions shape: {pos_np.shape}")
        print(f"[step_state] first particle: {pos_np[0]}")


# Optional small self-test (safe). This requires sim_state.py to exist.
if __name__ == "__main__":
    import yaml
    from src.simulation.sim_state import create_state

    cfg = yaml.safe_load(open("config.yaml"))
    state = create_state(cfg, device="cpu")  # change to "cuda" to test GPU

    print("Before:", state.positions.numpy()[0])
    step_state(state, cfg=cfg, debug_print=True)
    print("After :", state.positions.numpy()[0])
