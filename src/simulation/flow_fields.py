"""
Flow field models for Phase 4.

This module contains analytical flow profiles used to drive platelet motion.
Day 2 focus: cylindrical Poiseuille flow.
"""

from __future__ import annotations

import math


def poiseuille_velocity_radial(r: float, radius: float, vmax: float) -> float:
    """
    Return axial Poiseuille velocity at radial distance r.

    Formula:
        u(r) = vmax * (1 - (r / R)^2)

    where:
        r      = radial distance from vessel centerline
        radius = vessel radius R
        vmax   = maximum centerline velocity

    Returns 0 outside the vessel.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")

    if vmax < 0:
        raise ValueError("vmax must be non-negative")

    r_abs = abs(r)

    if r_abs >= radius:
        return 0.0

    q = r_abs / radius
    return vmax * (1.0 - q * q)


def radial_distance_yz(y: float, z: float, center_y: float = 0.0, center_z: float = 0.0) -> float:
    """
    Compute radial distance from vessel centerline in the y-z plane.
    Use this if flow direction is along x.
    """
    dy = y - center_y
    dz = z - center_z
    return math.sqrt(dy * dy + dz * dz)