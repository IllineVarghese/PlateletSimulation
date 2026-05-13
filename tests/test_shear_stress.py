import pytest

from src.simulation.shear_stress import (
    normalize_shear_stress,
    poiseuille_shear_rate_radial,
    poiseuille_shear_stress_radial,
    shear_stress_from_shear_rate,
)


def test_shear_rate_zero_at_center():
    shear_rate = poiseuille_shear_rate_radial(r=0.0, radius=1.0, vmax=2.0)
    assert shear_rate == pytest.approx(0.0)


def test_shear_rate_maximum_at_wall():
    shear_rate = poiseuille_shear_rate_radial(r=1.0, radius=1.0, vmax=2.0)
    assert shear_rate == pytest.approx(4.0)


def test_shear_rate_increases_toward_wall():
    radius = 1.0
    vmax = 2.0

    s_center = poiseuille_shear_rate_radial(0.0, radius, vmax)
    s_mid = poiseuille_shear_rate_radial(0.5, radius, vmax)
    s_wall = poiseuille_shear_rate_radial(1.0, radius, vmax)

    assert s_center < s_mid < s_wall


def test_shear_stress_from_shear_rate():
    stress = shear_stress_from_shear_rate(shear_rate=4.0, viscosity=0.0035)
    assert stress == pytest.approx(0.014)


def test_poiseuille_shear_stress_radial():
    stress = poiseuille_shear_stress_radial(
        r=1.0,
        radius=1.0,
        vmax=2.0,
        viscosity=0.0035,
    )
    assert stress == pytest.approx(0.014)


def test_normalize_shear_stress():
    value = normalize_shear_stress(shear_stress=0.5, reference_shear_stress=1.0)
    assert value == pytest.approx(0.5)


def test_normalize_shear_stress_clamps_to_one():
    value = normalize_shear_stress(shear_stress=2.0, reference_shear_stress=1.0)
    assert value == pytest.approx(1.0)