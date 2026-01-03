# src/warp_playground/warp_particles_cpu.py

from pathlib import Path
import numpy as np
import warp as wp


# -----------------------------
# Warp setup
# -----------------------------
wp.init()
DEVICE = "cpu"   # later on gaming laptop: change to "cuda"


# -----------------------------
# Simulation parameters
# -----------------------------
N = 500
DT = 0.01
STEPS = 200


# -----------------------------
# Initial state (NumPy -> Warp)
# -----------------------------
# initial positions in a small cube around origin
pos_np = (np.random.rand(N, 3).astype(np.float32) - 0.5) * 0.2

# velocities: constant flow in +z
vel_np = np.zeros((N, 3), dtype=np.float32)
vel_np[:, 2] = 0.2

pos = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
vel = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)


# -----------------------------
# Warp kernel
# -----------------------------
@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3),
              vel: wp.array(dtype=wp.vec3),
              dt: float):
    i = wp.tid()
    pos[i] = pos[i] + vel[i] * dt


# -----------------------------
# Main run
# -----------------------------
def main():
    # store full trajectory: (steps, N, 3)
    positions_steps = np.empty((STEPS, N, 3), dtype=np.float32)

    for s in range(STEPS):
        wp.launch(integrate, dim=N, inputs=[pos, vel, DT], device=DEVICE)

        # copy current positions to NumPy and store
        positions_steps[s] = pos.numpy()

    # save output
    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "positions_steps_warp.npy"
    np.save(out_path, positions_steps)

    # sanity checks
    last = positions_steps[-1]
    print("saved:", out_path.as_posix(), "shape:", positions_steps.shape)
    print("last step pos min:", last.min(axis=0))
    print("last step pos max:", last.max(axis=0))
    print("last step first 5:\n", last[:5])


if __name__ == "__main__":
    main()
