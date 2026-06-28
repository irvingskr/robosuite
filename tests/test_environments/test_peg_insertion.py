import numpy as np

from robosuite.models.objects import SquareHoleObject, SquarePegObject


def _collision_geoms(obj):
    return [geom for geom in obj.get_obj().iter("geom") if geom.get("group") == "0"]


def test_square_peg_object_contract():
    peg = SquarePegObject(name="peg")

    assert len(peg.joints) == 1
    assert set(peg.important_sites) >= {"center", "top", "bottom"}
    geoms = _collision_geoms(peg)
    assert len(geoms) == 1
    assert geoms[0].get("type") == "box"
    assert np.allclose(np.fromstring(geoms[0].get("size"), sep=" "), [0.02, 0.02, 0.05])


def test_square_hole_object_contract():
    hole = SquareHoleObject(name="hole")

    assert hole.joints == []
    assert set(hole.important_sites) >= {"mouth", "bottom", "axis"}
    geoms = _collision_geoms(hole)
    assert len(geoms) == 5
    assert np.allclose(hole.bottom_offset, [0.0, 0.0, 0.0])
    assert np.allclose(hole.top_offset, [0.0, 0.0, 0.065])
