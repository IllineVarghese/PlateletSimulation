from __future__ import annotations

import pytest

from src.simulation.cone_geometry import (
    cone_radius_at_position,
    inside_cone_vessel,
)


def test_cone_radius_interpolation():
    r0 = cone_radius_at_position(
        x=0.0,
        length=10.0,
        radius_start=1.0,
        radius_end=0.5,
    )

    r_mid = cone_radius_at_position(
        x=5.0,
        length=10.0,
        radius_start=1.0,
        radius_end=0.5,
    )

    r_end = cone_radius_at_position(
        x=10.0,
        length=10.0,
        radius_start=1.0,
        radius_end=0.5,
    )

    assert r0 == pytest.approx(1.0)
    assert r_mid == pytest.approx(0.75)
    assert r_end == pytest.approx(0.5)


def test_inside_cone_vessel():
    inside = inside_cone_vessel(
        x=2.0,
        y=0.2,
        z=0.2,
        length=10.0,
        radius_start=1.0,
        radius_end=0.5,
    )

    outside = inside_cone_vessel(
        x=9.0,
        y=1.0,
        z=1.0,
        length=10.0,
        radius_start=1.0,
        radius_end=0.5,
    )

    assert inside is True
    assert outside is False


def test_invalid_length():
    with pytest.raises(ValueError):
        cone_radius_at_position(
            x=0.0,
            length=0.0,
            radius_start=1.0,
            radius_end=0.5,
        )