from pathlib import Path
import numpy as np
import warp as wp

wp.init()
DEVICE = "cpu"   # later change to "cuda" on the gaming laptop

# -----------------------------
# Simulation parameters
# -----------------------------
N = 500
DT = 0.01
STEPS = 500

# Cone vessel along +Z
ZMIN = 0.0
ZMAX = 0.8

# Radius varies with z: R(z) = R0 + (R1 - R0) * t , t in [0,1]
R0 = 0.24   # radius at z=ZMIN
R1 = 0.12   # radius at z=ZMAX  (smaller -> stenosis-like)

# Particle size + collision
RADIUS = 0.01
RESTITUTION = 0.35
DAMP_TAN = 0.995

# Flow field (Poiseuille-like using local radius)
VMAX = 0.6

# Flow-through: respawn when exiting at ZMAX
RESPAWN = True


# -----------------------------
# Helpers (NumPy for init)
# -----------------------------
def r_of_z(z: np.ndarray) -> np.ndarray:
    t = (z - ZMIN) / (ZMAX - ZMIN)
    t = np.clip(t, 0.0, 1.0)
    return R0 + (R1 - R0) * t


def sample_inlet_positions(n: int) -> np.ndarray:
    theta = 2.0 * np.pi * np.random.rand(n).astype(np.float32)
    # sample near inlet (z just above ZMIN)
    z = np.full(n, ZMIN + RADIUS + 0.01, dtype=np.float32)
    Rin = r_of_z(z).astype(np.float32)
    # area-uniform inside (Rin - RADIUS)
    rr = (Rin - RADIUS) * np.sqrt(np.random.rand(n).astype(np.float32))
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


pos_np = sample_inlet_positions(N)
vel_np = np.zeros((N, 3), dtype=np.float32)

pos = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
vel = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)


# -----------------------------
# Warp functions/kernels
# -----------------------------
@wp.func
def cone_radius(z: float, zmin: float, zmax: float, r0: float, r1: float) -> float:
    t = (z - zmin) / (zmax - zmin)
    # clamp
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return r0 + (r1 - r0) * t


@wp.func
def poiseuille_vz(x: float, y: float, rv: float, vmax: float) -> float:
    # vmax at center, 0 at wall (approx)
    r2 = x * x + y * y
    t = 1.0 - (r2 / (rv * rv))
    if t < 0.0:
        t = 0.0
    return vmax * t


@wp.func
def hash01(seed: wp.uint32) -> float:
    # simple deterministic pseudo-rand in [0,1]
    v = seed
    v = v * wp.uint32(1664525) + wp.uint32(1013904223)
    return wp.float32(v & wp.uint32(0xFFFF)) / 65535.0


@wp.kernel
def step_cone_poiseuille(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
    zmin: float,
    zmax: float,
    r0: float,
    r1: float,
    pradius: float,
    restitution: float,
    damp_tan: float,
    vmax: float,
    respawn: int,
):
    i = wp.tid()

    p = pos[i]
    v = vel[i]

    # local vessel radius at current z
    rv = cone_radius(p[2], zmin, zmax, r0, r1)

    # flow velocity based on local radius
    vz = poiseuille_vz(p[0], p[1], rv, vmax)

    # damp lateral drift + enforce flow direction
    v = wp.vec3(v[0] * 0.98, v[1] * 0.98, vz)

    # integrate
    p = p + v * dt

    # enforce z bounds (soft)
    z_low = zmin + pradius
    z_high = zmax - pradius
    if p[2] < z_low:
        p = wp.vec3(p[0], p[1], z_low)

    # update local radius after moving
    rv = cone_radius(p[2], zmin, zmax, r0, r1)
    r_allowed = rv - pradius

    # --- cone wall collision ---
    x = p[0]
    y = p[1]
    r2 = x * x + y * y

    if r2 > r_allowed * r_allowed:
        r = wp.sqrt(r2) + 1.0e-12
        nx = x / r
        ny = y / r

        # project onto allowed radius at this z
        p = wp.vec3(nx * r_allowed, ny * r_allowed, p[2])

        # reflect lateral components
        vn = v[0] * nx + v[1] * ny
        vnx = vn * nx
        vny = vn * ny
        vtx = v[0] - vnx
        vty = v[1] - vny

        new_vx = (-vn * restitution) * nx + vtx * damp_tan
        new_vy = (-vn * restitution) * ny + vty * damp_tan

        # keep flow but damp slightly
        # recompute vz using projected position
        rv2 = cone_radius(p[2], zmin, zmax, r0, r1)
        new_vz = poiseuille_vz(p[0], p[1], rv2, vmax) * damp_tan

        v = wp.vec3(new_vx, new_vy, new_vz)

    # --- outlet handling ---
    if p[2] > z_high:
        if respawn == 1:
            # respawn at inlet with local inlet radius
            seed = wp.uint32(i) * wp.uint32(2654435761) + wp.uint32(12345)
            u1 = hash01(seed)
            u2 = hash01(seed + wp.uint32(99991))

            theta = 2.0 * 3.14159265 * u1

            rv_in = cone_radius(zmin + pradius + 0.01, zmin, zmax, r0, r1)
            rr = (rv_in - pradius) * wp.sqrt(u2)

            xnew = rr * wp.cos(theta)
            ynew = rr * wp.sin(theta)
            znew = zmin + pradius + 0.01

            p = wp.vec3(xnew, ynew, znew)
            v = wp.vec3(0.0, 0.0, poiseuille_vz(p[0], p[1], rv_in, vmax))
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
            step_cone_poiseuille,
            dim=N,
            inputs=[
                pos, vel, DT,
                ZMIN, ZMAX,
                R0, R1,
                RADIUS, RESTITUTION, DAMP_TAN,
                VMAX, int(RESPAWN)
            ],
            device=DEVICE,
        )
        positions_steps[s] = pos.numpy()

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "positions_steps_warp_cone_poiseuille.npy"
    np.save(out_path, positions_steps)

    last = positions_steps[-1]
    r = np.sqrt(last[:, 0] ** 2 + last[:, 1] ** 2)
    # worst-case allowed at each particle z
    allowed = r_of_z(last[:, 2]) - RADIUS
    print("saved:", out_path.as_posix(), "shape:", positions_steps.shape)
    print("max r:", float(r.max()), "max allowed:", float(allowed.max()))
    print("z min/max:", float(last[:, 2].min()), float(last[:, 2].max()))


if __name__ == "__main__":
    main()
