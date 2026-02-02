import numpy as np
import warp as wp

_WARP_INITIALIZED = False


@wp.kernel
def move_platelets(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt


def run_step(
    device="cpu",
    num_agents=10,
    dt=0.01,
    seed=42,
    geom_cfg=None,
    flow_cfg=None,
    debug_print=False,
):
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True

    # Choose device
    if device == "cuda" and wp.is_cuda_available():
        wp.set_device("cuda")
    else:
        wp.set_device("cpu")

    # Deterministic init
    rng = np.random.default_rng(seed)

    positions = wp.array(
        rng.random((num_agents, 3), dtype=np.float32),
        dtype=wp.vec3,
        device=wp.get_device(),
    )

    velocities = wp.array(
        (rng.standard_normal((num_agents, 3), dtype=np.float32) * 0.1),
        dtype=wp.vec3,
        device=wp.get_device(),
    )

    wp.launch(
        kernel=move_platelets,
        dim=num_agents,
        inputs=[positions, velocities, float(dt)],
    )

    # IMPORTANT: no huge prints
    if debug_print:
        pos_np = positions.numpy()
        print(f"[run_step] positions shape: {pos_np.shape}")
        print(f"[run_step] first particle: {pos_np[0]}")

    return positions


if __name__ == "__main__":
    run_step(device="cpu", num_agents=10, dt=0.01, seed=42, debug_print=True)
