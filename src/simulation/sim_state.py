# src/simulation/sim_state.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import warp as wp


@dataclass
class SimState:
    """
    Persistent simulation state (Week 2 core).

    The key rule:
    - Warp arrays here are allocated ONCE at the start.
    - Every timestep updates these arrays in-place.
    - We do NOT recreate positions/velocities every step.

    Fields:
    - positions: wp.array of wp.vec3 (agent positions)
    - velocities: wp.array of wp.vec3 (agent velocities)
    - activation: wp.array of float (optional scalar state per agent)
    - num_agents: number of particles/agents
    - device: wp.Device used for allocations/kernels
    - seed: RNG seed used for deterministic initialization
    - step_index: current timestep counter
    """
    positions: wp.array
    velocities: wp.array
    activation: wp.array
    num_agents: int
    device: wp.context.Device
    seed: int
    step_index: int = 0


def _get_cfg(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safe nested config getter.
    Example: _get_cfg(cfg, "simulation", "num_agents")
    """
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def create_state(cfg: Dict[str, Any], device: Optional[str] = None) -> SimState:
    """
    Create persistent state once (Week 2).

    This function:
    1) Initializes Warp
    2) Selects device
    3) Reads config values (num_agents, seed, geometry)
    4) Allocates Warp arrays ONCE
    5) Initializes positions/velocities deterministically

    Returns:
        SimState ready for timestep updates.
    """

    # 1) Initialize Warp runtime (safe to call multiple times)
    wp.init()

    # 2) Choose device
    # device can be None (auto), "cpu", or "cuda"
    if device is None:
        dev = wp.get_preferred_device()
    else:
        dev = wp.get_device(device)

    # 3) Read config values safely
    # Try both styles: cfg["simulation"]["num_agents"] or cfg["num_agents"]
    num_agents = _get_cfg(cfg, "simulation", "num_agents", default=None)
    if num_agents is None:
        num_agents = _get_cfg(cfg, "num_agents", default=None)
    if num_agents is None:
        raise ValueError("Config missing num_agents. Expected cfg['simulation']['num_agents'] or cfg['num_agents'].")

    seed = _get_cfg(cfg, "simulation", "seed", default=None)
    if seed is None:
        seed = _get_cfg(cfg, "seed", default=0)

    # Cylinder geometry defaults (adjust if your config uses different keys)
    radius = _get_cfg(cfg, "geometry", "cylinder", "radius", default=None)
    if radius is None:
        radius = _get_cfg(cfg, "geometry", "radius", default=1.0)

    length = _get_cfg(cfg, "geometry", "cylinder", "length", default=None)
    if length is None:
        length = _get_cfg(cfg, "geometry", "length", default=5.0)

    # 4) Allocate persistent Warp arrays ONCE
    # positions and velocities are vec3 arrays, activation is float array
    positions = wp.empty(num_agents, dtype=wp.vec3, device=dev)
    velocities = wp.empty(num_agents, dtype=wp.vec3, device=dev)
    activation = wp.zeros(num_agents, dtype=wp.float32, device=dev)

    # 5) Deterministic initialization on CPU using NumPy, then upload to Warp once
    # Why CPU init? It's simple and reproducible. We do it ONCE, so it won't hurt performance.

    rng = np.random.default_rng(int(seed))

    # Spawn points uniformly-ish inside cylinder cross-section:
    # - Sample r = R*sqrt(u) for uniform area density
    # - Sample theta uniform [0, 2pi)
    # - Sample z uniform [0, length)
    u = rng.random(num_agents, dtype=np.float64)
    r = radius * np.sqrt(u)
    theta = rng.random(num_agents, dtype=np.float64) * (2.0 * np.pi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.random(num_agents, dtype=np.float64) * float(length)

    # Pack into Nx3 float32
    pos_np = np.stack([x, y, z], axis=1).astype(np.float32)

    # Velocities start at 0 for Week 2 (Week 3 adds physical flow)
    vel_np = np.zeros((num_agents, 3), dtype=np.float32)

    # Upload once to Warp arrays
    positions.assign(wp.array(pos_np, dtype=wp.vec3, device=dev))
    velocities.assign(wp.array(vel_np, dtype=wp.vec3, device=dev))

    return SimState(
        positions=positions,
        velocities=velocities,
        activation=activation,
        num_agents=int(num_agents),
        device=dev,
        seed=int(seed),
        step_index=0,
    )
