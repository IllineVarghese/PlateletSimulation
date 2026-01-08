from pathlib import Path
import numpy as np
import warp as wp

wp.init()
DEVICE = "cuda:0"

# -----------------------------
# Params
# -----------------------------
N = 500
DT = 0.01
STEPS = 400

# Cylinder aligned with Z axis, centered at (0,0)
R_CYL = 0.2
ZMIN = 0.0
ZMAX = 0.8

# Particle collision params
PR = 0.01              # particle radius
REST = 0.6             # restitution (normal bounce)
DAMP_TAN = 0.995       # tangential damping (wall "friction")

# Flow: constant upward velocity (example)
FLOW_VZ = 0.35


# -----------------------------
# Initial state (inside cylinder)
# -----------------------------
def sample_inside_cylinder(n, r_cyl, zmin, zmax, pr):
    # uniform in area: r = sqrt(u) * (R - pr)
    u = np.random.rand(n).astype(np.float32)
    r = np.sqrt(u) * (r_cyl - pr) * 0.5  # start well inside
    theta = (2.0 * np.pi * np.random.rand(n)).astype(np.float32)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = (zmin + pr + (zmax - zmin - 2 * pr) * 0.25
         + (zmax - zmin) * 0.1 * np.random.rand(n)).astype(np.float32)
    return np.stack([x, y, z], axis=1).astype(np.float32)


pos_np = sample_inside_cylinder(N, R_CYL, ZMIN, ZMAX, PR)

vel_np = np.zeros((N, 3), dtype=np.float32)
vel_np[:, 2] = FLOW_VZ

pos = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
vel = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)


# -----------------------------
# Kernel: integrate + collide cylinder wall + caps
# -----------------------------
@wp.kernel
def integrate_and_collide_cylinder(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
    r_cyl: float,
    zmin: float,
    zmax: float,
    pr: float,
    rest: float,
    damp_tan: float,
):
    i = wp.tid()

    p = pos[i]
    v = vel[i]

    # integrate
    p = p + v * dt

    # Allowed interior limits
    r_allowed = r_cyl - pr
    z_low = zmin + pr
    z_high = zmax - pr

    # ---- side wall collision ----
    x = p[0]
    y = p[1]
    r2 = x * x + y * y
    r_allowed2 = r_allowed * r_allowed

    if r2 > r_allowed2:
        # normal is radial in xy plane
        r = wp.sqrt(r2)
        inv_r = 1.0 / (r + 1.0e-12)
        nx = x * inv_r
        ny = y * inv_r

        # Project position back to the wall
        p = wp.vec3(nx * r_allowed, ny * r_allowed, p[2])

        # Normal velocity component (in xy plane)
        vn = v[0] * nx + v[1] * ny

        # Only reflect if moving outward into the wall (vn > 0)
        if vn > 0.0:
            # Split velocity into normal and tangential components (xy only)
            vnx = vn * nx
            vny = vn * ny
            vtx = v[0] - vnx
            vty = v[1] - vny

            # Reflect normal component and damp tangential
            new_vx = (-rest * vn) * nx + damp_tan * vtx
            new_vy = (-rest * vn) * ny + damp_tan * vty
            new_vz = v[2]  # usually don't change vz on side wall

            v = wp.vec3(new_vx, new_vy, new_vz)

    # ---- bottom cap collision ----
    if p[2] < z_low:
        p = wp.vec3(p[0], p[1], z_low)
        if v[2] < 0.0:
            v = wp.vec3(damp_tan * v[0], damp_tan * v[1], -rest * v[2])

    # ---- top cap collision ----
    if p[2] > z_high:
        p = wp.vec3(p[0], p[1], z_high)
        if v[2] > 0.0:
            v = wp.vec3(damp_tan * v[0], damp_tan * v[1], -rest * v[2])

    pos[i] = p
    vel[i] = v


def main():
    positions_steps = np.empty((STEPS, N, 3), dtype=np.float32)

    for s in range(STEPS):
        wp.launch(
            integrate_and_collide_cylinder,
            dim=N,
            inputs=[pos, vel, DT, R_CYL, ZMIN, ZMAX, PR, REST, DAMP_TAN],
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

    # extra sanity: max radius should never exceed r_allowed by more than tiny epsilon
    r = np.sqrt(last[:, 0] ** 2 + last[:, 1] ** 2)
    print("last step max radius:", float(r.max()), "allowed:", float(R_CYL - PR))


if __name__ == "__main__":
    main()
