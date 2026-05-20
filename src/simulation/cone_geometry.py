from __future__ import annotations

import numpy as np


def cone_radius_at_position(
    x: float,
    length: float,
    radius_start: float,
    radius_end: float,
) -> float:
    """
    Linear cone radius interpolation.

    x = axial position
    """

    if length <= 0:
        raise ValueError("length must be positive")

    t = np.clip(x / length, 0.0, 1.0)

    return radius_start + t * (radius_end - radius_start)


def radial_distance(y: float, z: float) -> float:
    return np.sqrt(y**2 + z**2)


def inside_cone_vessel(
    x: float,
    y: float,
    z: float,
    length: float,
    radius_start: float,
    radius_end: float,
) -> bool:
    """
    Check if point lies inside cone vessel.
    """

    if x < 0 or x > length:
        return False

    r_local = cone_radius_at_position(
        x,
        length,
        radius_start,
        radius_end,
    )

    r = radial_distance(y, z)

    return bool(r <= r_local)