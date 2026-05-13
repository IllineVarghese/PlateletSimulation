from __future__ import annotations

import math


def poiseuille_shear_rate_radial(r: float, radius: float, vmax: float) -> float:
    """
    Shear rate magnitude for cylindrical Poiseuille flow.

    u(r) = vmax * (1 - (r/R)^2)
    du/dr = -2 * vmax * r / R^2

    shear_rate = |du/dr|
    """
    if radius <= 0:
        raise ValueError("radius must be positive")

    if vmax < 0:
        raise ValueError("vmax must be non-negative")

    r_abs = min(abs(r), radius)
    return abs(-2.0 * vmax * r_abs / (radius * radius))


def shear_stress_from_shear_rate(shear_rate: float, viscosity: float) -> float:
    """
    Convert shear rate to shear stress.

    tau = viscosity * shear_rate
    """
    if shear_rate < 0:
        raise ValueError("shear_rate must be non-negative")

    if viscosity <= 0:
        raise ValueError("viscosity must be positive")

    return viscosity * shear_rate


def poiseuille_shear_stress_radial(
    r: float,
    radius: float,
    vmax: float,
    viscosity: float,
) -> float:
    shear_rate = poiseuille_shear_rate_radial(r, radius, vmax)
    return shear_stress_from_shear_rate(shear_rate, viscosity)


def normalize_shear_stress(
    shear_stress: float,
    reference_shear_stress: float,
) -> float:
    """
    Normalize shear stress to [0, 1] for GRN input.

    This will later feed InShearStress.
    """
    if reference_shear_stress <= 0:
        raise ValueError("reference_shear_stress must be positive")

    value = shear_stress / reference_shear_stress
    return max(0.0, min(1.0, value))


def radial_distance_yz(y: float, z: float) -> float:
    return math.sqrt(y * y + z * z)