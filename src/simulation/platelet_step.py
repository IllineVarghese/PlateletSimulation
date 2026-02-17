# src/simulation/platelet_step.py

from __future__ import annotations

from typing import Any, Dict, Optional

import warp as wp

_WARP_INITIALIZED = False


@wp.kernel
def poiseuille_velocity_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    umax: float,
):
    """
    Set velocity to Poiseuille profile along +z:
        u(r) = umax * (1 - (r/R)^2)
    where r = sqrt(x^2 + y^2)
    """
    i = wp.tid()
    p = positions[i]

    r2 = p[0] * p[0] + p[1] * p[1]
    R2 = radius * radius

    # Outside cylinder: clamp flow to 0 (boundary handler will fix position later)
    u = 0.0
    if r2 < R2:
        u = umax * (1.0 - r2 / R2)

    velocities[i] = wp.vec3(0.0, 0.0, u)


@wp.kernel
def activation_kernel(
    positions: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    radius: float,
    near_wall_dist: float,
    activation_rate: float,
    dt: float,
):
    """
    Simple activation model:
    - If particle is within near_wall_dist of wall, activation increases.
    - Otherwise it slowly decays (very small decay).
    Activation is clamped to [0, 1].
    """
    i = wp.tid()
    p = positions[i]

    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    dist_to_wall = radius - r  # positive inside, ~0 near wall

    a = activation[i]

    if dist_to_wall <= near_wall_dist:
        a = a + activation_rate * dt
    else:
        # tiny decay so it doesn't stay permanently 1.0
        a = a - 0.05 * dt

    # clamp
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0

    activation[i] = a


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
    """
    Starter adhesion:
    - If near wall AND activation > threshold:
        reduce velocity by stick_factor (0 -> full stick, 1 -> no effect)
    """
    i = wp.tid()
    p = positions[i]

    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    dist_to_wall = radius - r

    if dist_to_wall <= near_wall_dist and activation[i] >= act_threshold:
        velocities[i] = velocities[i] * stick_factor


@wp.kernel
def integrate_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt


@wp.kernel
def cylinder_wall_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    restitution: float,
):
    """
    Enforce cylinder boundary at r=R.
    If outside, push back to the wall and damp radial velocity.

    This is a simple, stable boundary:
    - clamp x,y back onto radius
    - remove outward radial component of velocity (or reflect with restitution)
    """
    i = wp.tid()
    p = positions[i]
    v = velocities[i]

    x = p[0]
    y = p[1]
    r = wp.sqrt(x * x + y * y)

    if r > radius:
        # project position back to cylinder surface
        scale = radius / (r + 1e-8)
        x_new = x * scale
        y_new = y * scale
        p = wp.vec3(x_new, y_new, p[2])

        # compute outward normal
        nx = x_new / (radius + 1e-8)
        ny = y_new / (radius + 1e-8)

        # radial component of velocity
        vr = v[0] * nx + v[1] * ny

        # If moving outward, reflect/damp
        if vr > 0.0:
            v = v - (1.0 + restitution) * vr * wp.vec3(nx, ny, 0.0)

        positions[i] = p
        velocities[i] = v


@wp.kernel
def wrap_z_kernel(
    positions: wp.array(dtype=wp.vec3),
    length: float,
):
    """
    Keep z in [0, length) using periodic wrap.
    """
    i = wp.tid()
    p = positions[i]
    z = p[2]

    # wrap
    if z >= length:
        z = z - length * wp.floor(z / length)
    if z < 0.0:
        z = z + length * (1.0 + wp.floor((-z) / length))

    positions[i] = wp.vec3(p[0], p[1], z)


def step_state(
    state,
    dt: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
    debug_print: bool = False,
) -> None:
    """
    Week 3: physics step (Poiseuille + wall + activation + adhesion + integrate).

    Order:
    1) set flow velocities (Poiseuille)
    2) update activation based on near-wall
    3) apply adhesion (reduce velocity near wall if activated)
    4) integrate positions
    5) enforce cylinder boundary
    6) wrap z
    """

    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True

    # ---- read dt ----
    if dt is None:
        if cfg is not None:
            dt = cfg.get("simulation", {}).get("dt", cfg.get("dt", 0.001))
        else:
            dt = 0.001
    dt = float(dt)

    # ---- read geometry ----
    radius = 1.0
    length = 10.0
    if cfg is not None:
        g = cfg.get("geometry", {})
        radius = float(g.get("radius", radius))
        length = float(g.get("length", length))

    # ---- read flow ----
    umax = 1.0
    if cfg is not None:
        f = cfg.get("flow", {})
        umax = float(f.get("max_velocity", umax))

    # ---- activation / adhesion params (defaults; can later move to config) ----
    near_wall_dist = 0.10 * radius        # thinner near-wall band
    activation_rate = 0.5                 # slower activation
    act_threshold = 0.02                   # activation needed for adhesion
    stick_factor = 0.05                    # 0.0 = fully stuck, 1.0 = no adhesion
    restitution = 0.0                     # 0 = fully damp radial bounce, 1 = elastic
    if state.step_index == 0:
        print(
            "[params] near_wall_dist=", near_wall_dist,
            "act_threshold=", act_threshold,
            "stick_factor=", stick_factor
        )



    # 1) Set velocities from Poiseuille profile (GPU)
    wp.launch(
        poiseuille_velocity_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, radius, umax],
        device=state.device,
    )

    # 2) Update activation
    wp.launch(
        activation_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.activation, radius, near_wall_dist, activation_rate, dt],
        device=state.device,
    )

    # 3) Apply adhesion (reduce velocity if activated near wall)
    wp.launch(
        adhesion_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, state.activation, radius, near_wall_dist, act_threshold, stick_factor],
        device=state.device,
    )

    # 4) Integrate positions
    wp.launch(
        integrate_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, dt],
        device=state.device,
    )

    # 5) Enforce wall boundary
    wp.launch(
        cylinder_wall_kernel,
        dim=state.num_agents,
        inputs=[state.positions, state.velocities, radius, restitution],
        device=state.device,
    )

    # 6) Wrap z periodically
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
