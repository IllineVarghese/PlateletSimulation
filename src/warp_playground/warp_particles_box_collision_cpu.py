from pathlib import Path
import numpy as np
import warp as wp

wp.init()
DEVICE = "cuda:0"   # later change to "cuda"

# -----------------------------
# Params
# -----------------------------
N = 500
DT = 0.01
STEPS = 300

# Axis-aligned box bounds (min, max)
BMIN = (-0.2, -0.2, 0.0)
BMAX = ( 0.2,  0.2, 0.6)

# Particle radius and collision response
RADIUS = 0.01
RESTITUTION = 0.7   # 1.0 = perfectly elastic, <1.0 loses energy
DAMP_TANGENTIAL = 0.98  # slight damping along the wall

# -----------------------------
# Initial state
# -----------------------------
pos_np = (np.random.rand(N, 3).astype(np.float32) - 0.5) * 0.2
pos_np[:, 2] = pos_np[:, 2] * 0.2 + 0.1  # keep initial z inside box

vel_np = np.zeros((N, 3), dtype=np.float32)
vel_np[:, 2] = 0.3  # flow in +z

pos = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
vel = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)

bmin = wp.vec3(*BMIN)
bmax = wp.vec3(*BMAX)

# -----------------------------
# Kernel: integrate + collide with box
# -----------------------------
@wp.kernel
def integrate_and_collide(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
    bmin: wp.vec3,
    bmax: wp.vec3,
    radius: float,
    restitution: float,
    damp_tan: float,
):
    i = wp.tid()

    p = pos[i]
    v = vel[i]

    # integrate
    p = p + v * dt

    # helper: reflect component on collision
    # X min
    if p[0] < bmin[0] + radius:
        p = wp.vec3(bmin[0] + radius, p[1], p[2])
        v = wp.vec3(-v[0] * restitution, v[1] * damp_tan, v[2] * damp_tan)

    # X max
    if p[0] > bmax[0] - radius:
        p = wp.vec3(bmax[0] - radius, p[1], p[2])
        v = wp.vec3(-v[0] * restitution, v[1] * damp_tan, v[2] * damp_tan)

    # Y min
    if p[1] < bmin[1] + radius:
        p = wp.vec3(p[0], bmin[1] + radius, p[2])
        v = wp.vec3(v[0] * damp_tan, -v[1] * restitution, v[2] * damp_tan)

    # Y max
    if p[1] > bmax[1] - radius:
        p = wp.vec3(p[0], bmax[1] - radius, p[2])
        v = wp.vec3(v[0] * damp_tan, -v[1] * restitution, v[2] * damp_tan)

    # Z min
    if p[2] < bmin[2] + radius:
        p = wp.vec3(p[0], p[1], bmin[2] + radius)
        v = wp.vec3(v[0] * damp_tan, v[1] * damp_tan, -v[2] * restitution)

    # Z max
    if p[2] > bmax[2] - radius:
        p = wp.vec3(p[0], p[1], bmax[2] - radius)
        v = wp.vec3(v[0] * damp_tan, v[1] * damp_tan, -v[2] * restitution)

    pos[i] = p
    vel[i] = v


def main():
    positions_steps = np.empty((STEPS, N, 3), dtype=np.float32)

    for s in range(STEPS):
        wp.launch(
            integrate_and_collide,
            dim=N,
            inputs=[pos, vel, DT, bmin, bmax, RADIUS, RESTITUTION, DAMP_TANGENTIAL],
            device=DEVICE,
        )
        positions_steps[s] = pos.numpy()

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "positions_steps_warp_box.npy"
    np.save(out_path, positions_steps)

    last = positions_steps[-1]
    print("saved:", out_path.as_posix(), "shape:", positions_steps.shape)
    print("last step min:", last.min(axis=0))
    print("last step max:", last.max(axis=0))


if __name__ == "__main__":
    main()
