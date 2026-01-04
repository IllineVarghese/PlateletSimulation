from pathlib import Path
import numpy as np
import warp as wp

wp.init()
DEVICE = "cpu"   # later on gaming laptop: change to "cuda"

# -----------------------------
# Params
# -----------------------------
N = 500
DT = 0.01
STEPS = 400

# Cylinder (tube) aligned with Z axis:
# inside if x^2 + y^2 <= R^2 and z in [ZMIN, ZMAX]
R = 0.2
ZMIN = 0.0
ZMAX = 0.8

# Particle radius and collision response
RADIUS = 0.01
RESTITUTION = 0.6       # bounce on normal component
DAMP_TANGENTIAL = 0.995 # slight damping along wall

# Flow field: constant +z velocity (start simple)
FLOW_VZ = 0.35

# -----------------------------
# Initial state
# -----------------------------
# Sample positions inside a smaller radius so we start well inside the cylinder
theta = 2.0 * np.pi * np.random.rand(N).astype(np.float32)
rad = (0.08 * np.sqrt(np.random.rand(N))).astype(np.float32)  # <= 0.08
x = rad * np.cos(theta)
y = rad * np.sin(theta)
z = (ZMIN + 0.1 + 0.2 * np.random.rand(N)).astype(np.float32)

pos_np = np.stack([x, y, z], axis=1).astype(np.float32)

vel_np = np.zeros((N, 3), dtype=np.float32)
vel_np[:, 2] = FLOW_VZ

pos = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
vel = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)

# -----------------------------
# Kernel: integrate + collide with cylinder + z caps
# -----------------------------
@wp.kernel
def integrate_and_collide_cylinder(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
    cyl_r: float,
    zmin: float,
    zmax: float,
    pradius: float,
    restitution: float,
    damp_tan: float,
):
    i = wp.tid()

    p = pos[i]
    v = vel[i]

    # integrate
    p = p + v * dt

    # ---- collide with cylinder wall (radial) ----
    # wall is at r = cyl_r - pradius
    r_allowed = cyl_r - pradius

    x = p[0]
    y = p[1]
    r2 = x * x + y * y

    if r2 > r_allowed * r_allowed:
        r = wp.sqrt(r2) + 1.0e-12
        # normal pointing outward in xy plane
        nx = x / r
        ny = y / r

        # project point back to wall
        p = wp.vec3(nx * r_allowed, ny * r_allowed, p[2])

        # decompose velocity into normal/tangential parts in xy
        vn = v[0] * nx + v[1] * ny
        vnx = vn * nx
        vny = vn * ny
        vtx = v[0] - vnx
        vty = v[1] - vny

        # reflect normal, damp tangential; keep vz mostly unchanged (but apply mild damping)
        new_vx = (-vn * restitution) * nx + vtx * damp_tan
        new_vy = (-vn * restitution) * ny + vty * damp_tan
        new_vz = v[2] * damp_tan

        v = wp.vec3(new_vx, new_vy, new_vz)

    # ---- collide with z caps (optional; keeps particles in a finite tube) ----
    z_low = zmin + pradius
    z_high = zmax - pradius

    if p[2] < z_low:
        p = wp.vec3(p[0], p[1], z_low)
        v = wp.vec3(v[0] * damp_tan, v[1] * damp_tan, -v[2] * restitution)

    if p[2] > z_high:
        p = wp.vec3(p[0], p[1], z_high)
        v = wp.vec3(v[0] * damp_tan, v[1] * damp_tan, -v[2] * restitution)

    pos[i] = p
    vel[i] = v


def main():
    positions_steps = np.empty((STEPS, N, 3), dtype=np.float32)

    for s in range(STEPS):
        wp.launch(
            integrate_and_collide_cylinder,
            dim=N,
            inputs=[pos, vel, DT, R, ZMIN, ZMAX, RADIUS, RESTITUTION, DAMP_TANGENTIAL],
            device=DEVICE,
        )
        positions_steps[s] = pos.numpy()

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "positions_steps_warp_cyl.npy"
    np.save(out_path, positions_steps)

    last = positions_steps[-1]
    print("saved:", out_path.as_posix(), "shape:", positions_steps.shape)
    print("last step min:", last.min(axis=0))
    print("last step max:", last.max(axis=0))


if __name__ == "__main__":
    main()
