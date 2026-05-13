import pytest

from src.simulation.flow_fields import poiseuille_velocity_radial


def test_poiseuille_center_velocity_is_maximum():
    velocity = poiseuille_velocity_radial(r=0.0, radius=1.0, vmax=2.0)
    assert velocity == pytest.approx(2.0)


def test_poiseuille_wall_velocity_is_zero():
    velocity = poiseuille_velocity_radial(r=1.0, radius=1.0, vmax=2.0)
    assert velocity == pytest.approx(0.0)


def test_poiseuille_outside_wall_velocity_is_zero():
    velocity = poiseuille_velocity_radial(r=1.5, radius=1.0, vmax=2.0)
    assert velocity == pytest.approx(0.0)


def test_poiseuille_velocity_decreases_toward_wall():
    radius = 1.0
    vmax = 2.0

    u_center = poiseuille_velocity_radial(0.0, radius, vmax)
    u_mid = poiseuille_velocity_radial(0.5, radius, vmax)
    u_near_wall = poiseuille_velocity_radial(0.9, radius, vmax)

    assert u_center > u_mid > u_near_wall > 0.0


def test_poiseuille_rejects_invalid_radius():
    with pytest.raises(ValueError):
        poiseuille_velocity_radial(r=0.0, radius=0.0, vmax=2.0)