import numpy as np
import pytest

import robosuite as suite
from robosuite.models.objects import SquareHoleObject, SquarePegObject


def _collision_geoms(obj):
    return [geom for geom in obj.get_obj().iter("geom") if geom.get("group") == "0"]


def test_square_peg_object_contract():
    peg = SquarePegObject(name="peg")

    assert len(peg.joints) == 1
    assert set(peg.important_sites) >= {"center", "top", "bottom"}
    assert set(peg.important_sites.values()) <= set(peg.sites)
    geoms = _collision_geoms(peg)
    assert len(geoms) == 1
    assert geoms[0].get("type") == "box"
    assert np.allclose(np.fromstring(geoms[0].get("size"), sep=" "), [0.02, 0.02, 0.05])


def test_square_hole_object_contract():
    hole = SquareHoleObject(name="hole")

    assert hole.joints == []
    assert set(hole.important_sites) >= {"mouth", "bottom", "axis"}
    assert set(hole.important_sites.values()) <= set(hole.sites)
    geoms = _collision_geoms(hole)
    assert len(geoms) == 5
    assert np.allclose(hole.bottom_offset, [0.0, 0.0, 0.0])
    assert np.allclose(hole.top_offset, [0.0, 0.0, 0.065])


def _make_env(**kwargs):
    config = dict(
        env_name="PegInsertion",
        robots="Arx5",
        gripper_types="ArxGripper",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        hard_reset=False,
    )
    config.update(kwargs)
    return suite.make(**config)


def test_peg_insertion_is_registered_and_builds():
    assert "PegInsertion" in suite.ALL_ENVIRONMENTS
    env = _make_env()
    try:
        assert env.peg.root_body in env.sim.model.body_names
        assert env.hole.root_body in env.sim.model.body_names
    finally:
        env.close()


def test_peg_insertion_rejects_unsupported_robot():
    with pytest.raises(AssertionError, match="only supports Arx5"):
        _make_env(robots="Panda", gripper_types="ArxGripper")


def test_peg_insertion_rejects_unsupported_gripper():
    with pytest.raises(AssertionError, match="requires ArxGripper"):
        _make_env(gripper_types=None)


def _peg_contacts(env):
    peg_geoms = {env.sim.model.geom_name2id(name) for name in env.peg.contact_geoms}
    pairs = {
        frozenset((env.sim.data.contact[i].geom1, env.sim.data.contact[i].geom2))
        for i in range(env.sim.data.ncon)
    }
    return {
        "left": any(frozenset((env.left_finger_geom_id, peg_geom)) in pairs for peg_geom in peg_geoms),
        "right": any(frozenset((env.right_finger_geom_id, peg_geom)) in pairs for peg_geom in peg_geoms),
    }


def test_reset_places_peg_between_fingers_and_grasps_it():
    env = _make_env()
    try:
        env.reset()
        pad_midpoint = 0.5 * (
            env.sim.data.geom_xpos[env.left_finger_geom_id]
            + env.sim.data.geom_xpos[env.right_finger_geom_id]
        )
        peg_center = env.sim.data.site_xpos[env.peg_center_site_id]
        assert np.linalg.norm(peg_center[:2] - pad_midpoint[:2]) < 0.005
        assert peg_center[2] < pad_midpoint[2]

        action = np.zeros(env.action_dim)
        env.step(action)
        assert _peg_contacts(env) == {"left": True, "right": True}
    finally:
        env.close()


@pytest.mark.parametrize("requested_gripper", [-1.0, 0.0, 1.0])
def test_pre_action_forces_close_without_mutating_input(requested_gripper):
    env = _make_env()
    try:
        env.reset()
        action = np.zeros(env.action_dim)
        action[-1] = requested_gripper
        original = action.copy()
        env._pre_action(action, policy_step=True)
        assert np.array_equal(action, original)
        assert np.all(env.robots[0].gripper["right"].current_action < 0.0)
    finally:
        env.close()
