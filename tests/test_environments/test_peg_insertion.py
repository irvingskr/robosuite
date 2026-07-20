import numpy as np
import pytest

import robosuite as suite
import robosuite.environments.manipulation.peg_insertion as peg_module
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.peg_insertion import PEG_HALF_LENGTH
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
    visual_geoms = [geom for geom in peg.get_obj().iter("geom") if geom.get("group") == "1"]
    assert len(visual_geoms) == 1
    assert visual_geoms[0].get("material").endswith("peg_red")


def test_square_hole_object_contract():
    hole = SquareHoleObject(name="hole", movable=True)

    assert len(hole.joints) == 3
    assert hole.joints[0].endswith("slide_x")
    assert hole.joints[1].endswith("slide_y")
    assert hole.joints[2].endswith("yaw")
    assert set(hole.important_sites) >= {"mouth", "bottom", "axis"}
    assert set(hole.important_sites.values()) <= set(hole.sites)
    geoms = _collision_geoms(hole)
    assert len(geoms) == 5
    assert np.allclose(hole.bottom_offset, [0.0, 0.0, 0.0])
    assert np.allclose(hole.top_offset, [0.0, 0.0, 0.065])

    fixed_hole = SquareHoleObject(name="fixed_hole")
    assert fixed_hole.joints == []


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
        camera_id = env.sim.model.camera_name2id("agentview")
        assert np.allclose(env.sim.model.cam_pos[camera_id], peg_module.AGENTVIEW_POSITION)
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
        pad_mat = env.sim.data.geom_xmat[env.right_finger_geom_id].reshape(3, 3)
        expected_center = pad_midpoint + pad_mat[:, 0] * (
            peg_module.PEG_HALF_LENGTH - peg_module.PEG_GRASP_OVERLAP / 2.0
        )
        assert np.allclose(peg_center, expected_center, atol=1e-6)
        assert env._has_initial_peg_clearance_from_fk()

        action = np.zeros(env.action_dim)
        env.step(action)
        assert _peg_contacts(env) == {"left": True, "right": True}
    finally:
        env.close()


def test_randomized_reset_rejects_joint_limit_clipping():
    env = _make_env(
        seed=7201,
        initialization_noise={
            "type": "uniform",
            "magnitude": [0.0, 0.0, 0.0, 0.4, 0.4, 0.4],
        },
    )
    try:
        for _ in range(20):
            env.reset()
            robot = env.robots[0]
            qpos = env.sim.data.qpos[robot._ref_joint_pos_indexes]
            joint_ids = robot._ref_joint_indexes
            ranges = env.sim.model.jnt_range[joint_ids]
            limited = env.sim.model.jnt_limited[joint_ids].astype(bool)
            assert np.all(qpos[limited] > ranges[limited, 0] + 1e-4)
            assert np.all(qpos[limited] < ranges[limited, 1] - 1e-4)
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


def test_fixed_hole_position_is_restored():
    env = _make_env(randomize_hole_position=False)
    try:
        env.reset()
        first = env.sim.data.body_xpos[env.hole_body_id].copy()
        assert env.hole_planar_joints == ()
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        after_steps = env.sim.data.body_xpos[env.hole_body_id].copy()
        env.reset()
        second = env.sim.data.body_xpos[env.hole_body_id].copy()
        assert np.allclose(first[:2], peg_module.FIXED_HOLE_XY)
        assert np.allclose(after_steps, first)
        assert np.allclose(second[:2], peg_module.FIXED_HOLE_XY)
    finally:
        env.close()


def test_random_hole_position_is_seeded_and_in_circle():
    radius = 0.10
    env1 = _make_env(seed=7, randomize_hole_position=True, hole_position_radius=radius)
    env2 = _make_env(seed=7, randomize_hole_position=True, hole_position_radius=radius)
    try:
        sequence1 = []
        sequence2 = []
        for _ in range(3):
            env1.reset()
            env2.reset()
            sequence1.append(env1.sim.data.body_xpos[env1.hole_body_id, :2].copy())
            sequence2.append(env2.sim.data.body_xpos[env2.hole_body_id, :2].copy())
        assert np.allclose(sequence1, sequence2)
        assert all(np.linalg.norm(xy - peg_module.FIXED_HOLE_XY) <= radius + 1e-9 for xy in sequence1)
        assert not np.allclose(sequence1[0], sequence1[1])
        hole_mat = env1.sim.data.body_xmat[env1.hole_body_id].reshape(3, 3)
        yaw = np.arctan2(hole_mat[1, 0], hole_mat[0, 0])
        assert -peg_module.HOLE_YAW_RANGE <= yaw <= peg_module.HOLE_YAW_RANGE
        assert env1.hole_planar_joints == ()
    finally:
        env1.close()
        env2.close()


def test_randomized_hole_stays_static_under_external_force():
    env = _make_env(randomize_hole_position=True)
    try:
        env.reset()
        initial_position = env.sim.data.body_xpos[env.hole_body_id].copy()
        initial_rotation = env.sim.data.body_xmat[env.hole_body_id].copy()

        for _ in range(200):
            env.sim.data.xfrc_applied[env.hole_body_id, 0] = 50.0
            env.sim.step()
        env.sim.data.xfrc_applied[env.hole_body_id] = 0.0

        assert np.allclose(env.sim.data.body_xpos[env.hole_body_id], initial_position)
        assert np.allclose(env.sim.data.body_xmat[env.hole_body_id], initial_rotation)
    finally:
        env.close()


def test_randomized_hole_is_one_rigid_structure():
    env = _make_env(randomize_hole_position=True)
    try:
        env.reset()
        geom_body_ids = {
            env.sim.model.geom_bodyid[env.sim.model.geom_name2id(name)]
            for name in env.hole.contact_geoms
        }
        assert geom_body_ids == {env.hole_body_id}
        assert env.hole_planar_joints == ()
    finally:
        env.close()


def _set_peg_pose(env, depth=0.04, xy_error=0.0, roll=0.0, yaw=0.0):
    hole_mat = env.sim.data.body_xmat[env.hole_body_id].reshape(3, 3)
    peg_mat = hole_mat @ T.euler2mat(np.array([roll, 0.0, yaw]))
    quat_xyzw = T.mat2quat(peg_mat)
    peg_axis = peg_mat[:, 2]
    mouth = env.sim.data.site_xpos[env.hole_mouth_site_id].copy()
    bottom = mouth + xy_error * hole_mat[:, 0] - depth * hole_mat[:, 2]
    center = bottom + PEG_HALF_LENGTH * peg_axis
    env.sim.data.set_joint_qpos(
        env.peg_joint,
        np.concatenate([center, T.convert_quat(quat_xyzw, to="wxyz")]),
    )
    env.sim.forward()


@pytest.mark.parametrize(
    "pose, expected",
    [
        ({"depth": 0.041}, True),
        ({"depth": 0.039}, False),
        ({"xy_error": 0.004}, False),
        ({"depth": 0.041, "roll": np.deg2rad(6.0)}, True),
        ({"depth": 0.041, "yaw": np.deg2rad(6.0)}, True),
        ({"depth": 0.041, "yaw": np.deg2rad(90.0)}, True),
    ],
)
def test_success_boundaries(pose, expected):
    env = _make_env()
    try:
        env.reset()
        _set_peg_pose(env, **pose)
        assert env._check_success() is expected
    finally:
        env.close()


def test_sparse_and_dense_rewards_are_scaled_and_delta_based():
    sparse = _make_env(reward_shaping=False, reward_scale=2.0)
    dense = _make_env(reward_shaping=True, reward_scale=2.0)
    try:
        sparse.reset()
        dense.reset()
        _set_peg_pose(sparse, depth=0.041)
        _set_peg_pose(dense, depth=0.041)
        assert sparse.reward() == pytest.approx(2.0)
        assert dense.reward() == pytest.approx(2.0)

        _set_peg_pose(sparse, depth=0.02)
        _set_peg_pose(dense, depth=0.02)
        action = np.zeros(dense.action_dim)
        assert sparse.reward() == 0.0
        reward, _, _ = dense._post_action(action)
        assert reward == 0.0

        _set_peg_pose(dense, depth=0.03)
        assert dense.reward() > 0.0
        assert dense.reward() > 0.0
        reward, _, _ = dense._post_action(action)
        assert reward > 0.0
        reward, _, _ = dense._post_action(action)
        assert reward == 0.0

        _set_peg_pose(dense, depth=0.01)
        reward, _, _ = dense._post_action(action)
        assert reward < 0.0
    finally:
        sparse.close()
        dense.close()


def test_insertion_potential_rewards_depth_when_aligned():
    env = _make_env(reward_shaping=True)
    try:
        env.reset()
        _set_peg_pose(env, depth=0.005, xy_error=0.0)
        shallow = env._reward_potential()
        _set_peg_pose(env, depth=0.025, xy_error=0.0)
        deeper = env._reward_potential()
        assert deeper > shallow
    finally:
        env.close()


def test_insertion_potential_penalizes_depth_when_misaligned():
    env = _make_env(reward_shaping=True)
    try:
        env.reset()
        _set_peg_pose(env, depth=0.005, xy_error=0.02)
        shallow = env._reward_potential()
        _set_peg_pose(env, depth=0.025, xy_error=0.02)
        deeper = env._reward_potential()
        assert deeper < shallow
    finally:
        env.close()


def test_staged_insertion_reward_uses_soft_alignment_gate():
    env = _make_env(reward_shaping=True)
    try:
        env.reset()
        _set_peg_pose(env, depth=0.005, xy_error=0.012)
        _, shallow_alignment, shallow_insertion = env.staged_rewards()
        _set_peg_pose(env, depth=0.025, xy_error=0.012)
        _, deeper_alignment, deeper_insertion = env.staged_rewards()

        assert shallow_insertion > 0.0
        assert shallow_insertion >= shallow_alignment
        assert deeper_insertion > deeper_alignment
        assert deeper_insertion > shallow_insertion
    finally:
        env.close()


def test_object_observations_have_expected_names_and_shapes():
    env = _make_env()
    try:
        obs = env.reset()
        expected = {
            "peg_pos": (3,),
            "peg_quat": (4,),
            "hole_pos": (3,),
            "peg_to_hole_pos": (3,),
            "peg_bottom_to_hole_pos": (3,),
            "insertion_depth": (),
            "xy_error": (),
            "vertical_angle": (),
            "yaw_error": (),
        }
        for name, shape in expected.items():
            assert name in obs
            assert np.asarray(obs[name]).shape == shape
        assert "object-state" in obs
    finally:
        env.close()


def test_peg_insertion_zero_action_smoke():
    env = _make_env()
    try:
        obs = env.reset()
        for _ in range(10):
            obs, reward, done, info = env.step(np.zeros(env.action_dim))
            assert np.isfinite(reward)
            assert all(np.all(np.isfinite(value)) for value in obs.values())
            assert isinstance(done, (bool, np.bool_))
            assert isinstance(info, dict)
    finally:
        env.close()
