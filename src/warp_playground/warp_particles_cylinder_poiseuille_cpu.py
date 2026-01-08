from pathlib import Path
import numpy as np
import warp as wp

wp.init()
DEVICE = "cuda:0"  # later: "cuda"

# -----------------------------
# Params
# -----------------------------
N = 500
DT = 0.01
STEPS = 500

# Cylinder vessel
R = 0.2
ZMIN = 0.0
ZMAX = 0.8

# Particle radius + wall response
RADIUS = 0.01
RESTITUTION = 0.4
DAMP_TAN = 0.995

# Poiseuille flow: vz = VMAX*(1 - (r/R)^2)
VMAX = 0.6

# Inlet/Outlet behaviour
# Instead of bouncing at ZMAX, we "respawn" at inlet (more realistic flow-through)
RESPAWN = True


def sample_inlet_positions(n: int) -> np.ndarray:
    theta = 2.0 * np.pi * np.random.rand(n).astype(np.float32)
    # sample radius inside (R - RADIUS) with area-uniform distribution
    rr = (R - RADIUS) * np.sqrt(np.random.rand(n).astype(np.float32))
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    z = np.full(n, ZMIN + RADIUS + 0.01, dtype=np.float32)  # slightly inside
    return np.stack([x, y, z], axis=1).astype(np.float32)


pos_np = sample_inlet_positions(N)
vel_np = np.zeros((N, 3), dtype=np.float32)

pos = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
vel = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)


@wp.func
def poiseuille_vz(x: float, y: float, Rv: float, vmax: float) -> float:
    r2 = x * x + y * y
    # clamp to avoid negative if slightly outside due to numeric issues
    t = 1.0 - (r2 / (Rv * Rv))
    if t < 0.0:
        t = 0.0
    return vmax * t


@wp.kernel
def step_cylinder_poiseuille(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
    Rv: float,
    zmin: float,
    zmax: float,
    pradius: float,
    restitution: float,
    damp_tan: float,
    vmax: float,
    respawn: int,
):
    i = wp.tid()

    p = pos[i]
    v = vel[i]

    # compute flow velocity at current radial position (x,y)
    vz = poiseuille_vz(p[0], p[1], Rv, vmax)
    v = wp.vec3(v[0] * 0.98, v[1] * 0.98, vz)  # decay lateral drift, enforce vz field

    # integrate
    p = p + v * dt

    # --- cylinder wall collision ---
    r_allowed = Rv - pradius
    x = p[0]
    y = p[1]
    r2 = x * x + y * y

    if r2 > r_allowed * r_allowed:
        r = wp.sqrt(r2) + 1.0e-12
        nx = x / r
        ny = y / r

        # project back to wall
        p = wp.vec3(nx * r_allowed, ny * r_allowed, p[2])

        vn = v[0] * nx + v[1] * ny
        vnx = vn * nx
        vny = vn * ny
        vtx = v[0] - vnx
        vty = v[1] - vny

        new_vx = (-vn * restitution) * nx + vtx * damp_tan
        new_vy = (-vn * restitution) * ny + vty * damp_tan
        # keep flow direction but damp slightly
        new_vz = v[2] * damp_tan

        v = wp.vec3(new_vx, new_vy, new_vz)

    # --- z handling ---
    z_low = zmin + pradius
    z_high = zmax - pradius

    if p[2] < z_low:
        p = wp.vec3(p[0], p[1], z_low)

    if p[2] > z_high:
        if respawn == 1:
            # respawn at inlet with random (x,y) inside radius
            # Warp random: simple hash-based pseudo randomness using i and timestep-like p.z
            # (good enough for a demo)
            seed = wp.uint32(i) * wp.uint32(1664525) + wp.uint32(1013904223)
            u1 = wp.float32((seed & wp.uint32(0xFFFF)))/65535.0
            seed = seed * wp.uint32(1664525) + wp.uint32(1013904223)
            u2 = wp.float32((seed & wp.uint32(0xFFFF)))/65535.0

            theta = 2.0 * 3.14159265 * u1
            rr = (Rv - pradius) * wp.sqrt(u2)

            p = wp.vec3(rr * wp.cos(theta), rr * wp.sin(theta), zmin + pradius + 0.01)
            # reset lateral velocity
            v = wp.vec3(0.0, 0.0, poiseuille_vz(p[0], p[1], Rv, vmax))
        else:
            # bounce at end-cap
            p = wp.vec3(p[0], p[1], z_high)
            v = wp.vec3(v[0] * damp_tan, v[1] * damp_tan, -v[2] * restitution)

    pos[i] = p
    vel[i] = v


def main():
    positions_steps = np.empty((STEPS, N, 3), dtype=np.float32)

    for s in range(STEPS):
        wp.launch(
            step_cylinder_poiseuille,
            dim=N,
            inputs=[pos, vel, DT, R, ZMIN, ZMAX, RADIUS, RESTITUTION, DAMP_TAN, VMAX, int(RESPAWN)],
            device=DEVICE,
        )
        positions_steps[s] = pos.numpy()

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "positions_steps_warp_cyl_poiseuille.npy"
    np.save(out_path, positions_steps)

    last = positions_steps[-1]
    r = np.sqrt(last[:, 0] ** 2 + last[:, 1] ** 2)
    print("saved:", out_path.as_posix(), "shape:", positions_steps.shape)
    print("max r:", float(r.max()), "should be <=", R - RADIUS)
    print("z min/max:", float(last[:, 2].min()), float(last[:, 2].max()))


if __name__ == "__main__":
    main()
