from __future__ import annotations

from typing import Any, Dict, Optional

import warp as wp

_WARP_INITIALIZED = False


# ============================================================
# FLOW
# ============================================================

@wp.kernel
def poiseuille_velocity_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    umax: float,
):
    i = wp.tid()
    p = positions[i]

    r2 = p[0] * p[0] + p[1] * p[1]
    R2 = radius * radius

    u = 0.0
    if r2 < R2:
        u = umax * (1.0 - r2 / R2)

    velocities[i] = wp.vec3(0.0, 0.0, u)


# ============================================================
# ACTIVATION
# ============================================================

@wp.kernel
def activation_kernel(
    positions: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    radius: float,
    near_wall_dist: float,
    activation_rate: float,
    decay_rate: float,
    dt: float,
):
    i = wp.tid()
    p = positions[i]

    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    dist_to_wall = radius - r

    a = activation[i]

    if dist_to_wall <= near_wall_dist:
        a = a + activation_rate * dt
    else:
        a = a - decay_rate * dt

    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0

    activation[i] = a


# ============================================================
# ADHESION
# ============================================================

@wp.kernel
def adhesion_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    radius: float,
    near_wall_dist: float,
    act_threshold: float,
    stick_factor: float,
):
    i = wp.tid()
    p = positions[i]

    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    dist_to_wall = radius - r

    if dist_to_wall <= near_wall_dist and activation[i] >= act_threshold:
        velocities[i] = velocities[i] * stick_factor


# ============================================================
# INTEGRATION
# ============================================================

@wp.kernel
def integrate_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt


# ============================================================
# WALL
# ============================================================

@wp.kernel
def cylinder_wall_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    restitution: float,
):
    i = wp.tid()
    p = positions[i]
    v = velocities[i]

    x = p[0]
    y = p[1]
    r = wp.sqrt(x * x + y * y)

    if r > radius:
        scale = radius / (r + 1e-8)
        x_new = x * scale
        y_new = y * scale
        p = wp.vec3(x_new, y_new, p[2])

        nx = x_new / (radius + 1e-8)
        ny = y_new / (radius + 1e-8)

        vr = v[0] * nx + v[1] * ny

        if vr > 0.0:
            v = v - (1.0 + restitution) * vr * wp.vec3(nx, ny, 0.0)

        positions[i] = p
        velocities[i] = v


# ============================================================
# PERIODIC Z
# ============================================================

@wp.kernel
def wrap_z_kernel(
    positions: wp.array(dtype=wp.vec3),
    length: float,
):
    i = wp.tid()
    p = positions[i]
    z = p[2]

    if z >= length:
        z = z - length * wp.floor(z / length)
    if z < 0.0:
        z = z + length * (1.0 + wp.floor((-z) / length))

    positions[i] = wp.vec3(p[0], p[1], z)


# ============================================================
# MAIN STEP
# ============================================================

def step_state(
    state,
    dt: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
    debug_print: bool = False,
) -> None:

    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True

    # ---------------- DT ----------------
    if dt is None:
        if cfg is not None:
            dt = cfg.get("simulation", {}).get("dt", 0.001)
        else:
            dt = 0.001
    dt = float(dt)

    # ---------------- GEOMETRY ----------------
    radius = 1.0
    length = 10.0
    if cfg is not None:
        g = cfg.get("geometry", {})
        radius = float(g.get("radius", radius))
        length = float(g.get("length", length))

    # ---------------- FLOW ----------------
    umax = 1.0
    if cfg is not None:
        f = cfg.get("flow", {})
        umax = float(f.get("max_velocity", umax))

    # ---------------- ACTIVATION PARAMS ----------------
    act_cfg = cfg.get("activation", {}) if cfg else {}
    near_frac = act_cfg.get("near_wall_dist_frac", 0.10)
    activation_rate = act_cfg.get("activation_rate", 0.5)
    decay_rate = act_cfg.get("decay_rate", 0.05)

    near_wall_dist = near_frac * radius

    # ---------------- ADHESION PARAMS ----------------
    adh_cfg = cfg.get("adhesion", {}) if cfg else {}
    adh_enabled = adh_cfg.get("enabled", True)
    act_threshold = adh_cfg.get("act_threshold", 0.02)
    stick_factor = adh_cfg.get("stick_factor", 1.0)

    # ---------------- WALL ----------------
    wall_cfg = cfg.get("wall", {}) if cfg else {}
    restitution = wall_cfg.get("restitution", 0.0)

    if state.step_index == 0:
        print(
            "[params]",
            "near_wall_dist=", near_wall_dist,
            "act_threshold=", act_threshold,
            "stick_factor=", stick_factor,
            "adh_enabled=", adh_enabled
        )

    # 1) Flow
    wp.launch(
        poiseuille_velocity_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, radius, umax],
        device=state.device,
    )

    # 2) Activation
    wp.launch(
        activation_kernel,
        dim=state.num_agents,
        inputs=[
            state.positions,
            state.activation,
            radius,
            near_wall_dist,
            activation_rate,
            decay_rate,
            dt,
        ],
        device=state.device,
    )

    # 3) Adhesion (optional)
    if adh_enabled:
        wp.launch(
            adhesion_kernel,
            dim=state.num_agents,
            inputs=[
                state.positions,
                state.velocities,
                state.activation,
                radius,
                near_wall_dist,
                act_threshold,
                stick_factor,
            ],
            device=state.device,
        )

    # 4) Integrate
    wp.launch(
        integrate_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, dt],
        device=state.device,
    )

    # 5) Wall
    wp.launch(
        cylinder_wall_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, radius, restitution],
        device=state.device,
    )

    # 6) Wrap Z
    wp.launch(
        wrap_z_kernel,
        dim=state.num_agents,
        inputs=[state.positions, length],
        device=state.device,
    )

    state.step_index += 1

    if debug_print:
        pos_np = state.positions.numpy()
        act_np = state.activation.numpy()
        print(f"[step_state] step={state.step_index} pos[0]={pos_np[0]} act[0]={act_np[0]:.3f}")