from types import SimpleNamespace

import numpy as np
import pytest

import robosuite as suite
from robosuite.environments.manipulation.stack import Stack, StackRewardStage


def _stage_kwargs(**overrides):
    values = dict(
        distance=0.0,
        red_left_contact=True,
        red_right_contact=True,
        red_grasped=True,
        green_grasped=False,
        red_height=0.82,
        table_height=0.8,
        horizontal_distance=0.20,
        target_height_error=0.05,
        success=False,
    )
    values.update(overrides)
    return values


def _reward_only_stack(
    *,
    snapshot=None,
    previous=0.0,
    previous_stage=StackRewardStage.APPROACH,
    reward_shaping=True,
    reward_scale=None,
    previous_red_grasped=False,
    previous_green_grasped=False,
    previous_green_shift=0.0,
):
    env = Stack.__new__(Stack)
    env.reward_shaping = reward_shaping
    env.reward_scale = reward_scale
    env._reward_stage = previous_stage
    env._prev_reward_potential = previous
    env._prev_red_grasped = previous_red_grasped
    env._prev_green_grasped = previous_green_grasped
    env._prev_green_effective_shift = previous_green_shift
    current = _stage_kwargs(
        distance=10.0,
        red_left_contact=False,
        red_right_contact=False,
        red_grasped=False,
    )
    current["green_effective_shift"] = 0.0
    if snapshot:
        current.update(snapshot)
    env._snapshot_data = current
    env._reward_snapshot = lambda: dict(env._snapshot_data)
    env.timestep = 1
    env.horizon = 100
    env.ignore_done = False
    env.done = False
    return env


def test_shaped_reward_adds_sparse_reward_and_discounted_potential_difference():
    env = _reward_only_stack(
        snapshot={"success": True},
        previous=1.70,
        previous_stage=StackRewardStage.PLACE,
        previous_red_grasped=True,
    )

    reward = env.reward(action=None)

    assert reward == pytest.approx(2.0 + 0.99 * 2.0 - 1.70)
    assert env._prev_reward_potential == 1.70


def test_reward_potential_is_zero_without_progress_and_positive_at_success():
    env = _reward_only_stack()

    assert env._reward_potential() == 0.0

    env._snapshot_data["success"] = True
    assert env._reward_potential() == pytest.approx(2.0)


def test_read_only_potential_does_not_advance_stage_without_transition():
    env = _reward_only_stack(
        snapshot={
            "red_grasped": True,
            "red_left_contact": True,
            "red_right_contact": True,
            "red_height": 0.92,
            "horizontal_distance": 0.20,
        },
        previous=0.95,
        previous_stage=StackRewardStage.LIFT,
        previous_red_grasped=True,
    )

    assert env._reward_potential() == pytest.approx(0.95)
    assert env._reward_stage is StackRewardStage.LIFT


def test_staged_rewards_keeps_success_separate_from_progress_components():
    env = _reward_only_stack(
        snapshot={"success": True, "distance": 0.0},
        previous=1.70,
        previous_stage=StackRewardStage.PLACE,
        previous_red_grasped=True,
    )

    approach, active_stage, success = env.staged_rewards()

    assert approach == pytest.approx(0.30)
    assert active_stage == 0.0
    assert success == 2.0


def test_sparse_reward_and_existing_scaling_are_preserved_without_shaping():
    env = _reward_only_stack(
        snapshot={"success": True},
        previous=1.70,
        previous_stage=StackRewardStage.PLACE,
        reward_shaping=False,
        reward_scale=3.0,
    )

    assert env.reward(action=None) == pytest.approx(3.0)
    assert env._prev_reward_potential == 1.70


def test_post_action_updates_potential_once_and_stalling_has_discount_cost():
    env = _reward_only_stack(
        snapshot={
            "red_grasped": True,
            "red_left_contact": True,
            "red_right_contact": True,
            "red_height": 0.92,
            "horizontal_distance": 0.10,
        },
        previous=0.95,
        previous_stage=StackRewardStage.ALIGN,
        previous_red_grasped=True,
    )
    potential = Stack._stage_potential(StackRewardStage.ALIGN, **env._snapshot_data)

    first_reward, first_done, first_info = env._post_action(action=None)
    second_reward, second_done, second_info = env._post_action(action=None)

    assert first_reward == pytest.approx(0.99 * potential - 0.95)
    assert second_reward == pytest.approx(0.99 * potential - potential)
    assert first_reward > 0.0
    assert second_reward < 0.0
    assert env._reward_stage is StackRewardStage.ALIGN
    assert env._prev_reward_potential == pytest.approx(potential)
    assert env._prev_red_grasped is True
    assert env._prev_green_grasped is False
    assert env._prev_green_effective_shift == 0.0
    assert first_done is False
    assert second_done is False
    assert first_info == {}
    assert second_info == {}


def test_grasp_acquisition_bonus_is_one_time_and_read_only_calls_are_idempotent():
    env = _reward_only_stack(
        snapshot={
            "red_grasped": True,
            "red_left_contact": True,
            "red_right_contact": True,
        },
        previous=0.30,
    )
    expected = 0.99 * 0.50 - 0.30 + 0.35

    assert env.reward(None) == pytest.approx(expected)
    assert env.reward(None) == pytest.approx(expected)
    assert env._reward_stage is StackRewardStage.APPROACH
    assert env._prev_reward_potential == pytest.approx(0.30)
    assert env._prev_red_grasped is False


def test_uninitialized_grasp_history_skips_event_bonus_but_keeps_pbrs_delta():
    env = _reward_only_stack(
        snapshot={
            "red_grasped": True,
            "red_left_contact": True,
            "red_right_contact": True,
        },
        previous=0.30,
        previous_red_grasped=None,
    )

    assert env.reward(None) == pytest.approx(0.99 * 0.50 - 0.30)


def test_reward_scale_applies_to_pbrs_and_grasp_acquisition_together():
    env = _reward_only_stack(
        snapshot={
            "red_grasped": True,
            "red_left_contact": True,
            "red_right_contact": True,
        },
        previous=0.30,
        reward_scale=3.0,
    )
    unscaled = 0.99 * 0.50 - 0.30 + 0.35

    assert env.reward(None) == pytest.approx(unscaled * 3.0 / 2.0)


def test_post_action_consumes_grasp_acquisition_once():
    env = _reward_only_stack(
        snapshot={
            "red_grasped": True,
            "red_left_contact": True,
            "red_right_contact": True,
        },
        previous=0.30,
    )
    expected = 0.99 * 0.50 - 0.30 + 0.35

    first_reward = env._post_action(action=None)[0]
    assert first_reward == pytest.approx(expected)
    assert env._reward_stage is StackRewardStage.LIFT
    assert env._prev_reward_potential == pytest.approx(0.50)
    assert env._prev_red_grasped is True

    second_reward = env._post_action(action=None)[0]
    assert second_reward == pytest.approx(0.99 * 0.50 - 0.50)


def test_invalid_drop_is_penalized_but_successful_release_is_not():
    dropped = _reward_only_stack(
        snapshot={"distance": 0.0},
        previous=0.60,
        previous_stage=StackRewardStage.LIFT,
        previous_red_grasped=True,
    )
    successful = _reward_only_stack(
        snapshot={"success": True},
        previous=1.70,
        previous_stage=StackRewardStage.PLACE,
        previous_red_grasped=True,
    )

    assert dropped.reward(None) == pytest.approx(0.99 * 0.30 - 0.60 - 0.45)
    assert successful.reward(None) == pytest.approx(2.0 + 0.99 * 2.0 - 1.70)


def test_complete_grasp_drop_event_pair_is_negative():
    acquire = Stack._grasp_event_reward(False, True, False)
    drop = Stack._grasp_event_reward(True, False, False)

    assert acquire == pytest.approx(0.35)
    assert drop == pytest.approx(-0.45)
    assert acquire + drop == pytest.approx(-0.10)


def test_no_progress_rollout_does_not_accumulate_positive_shaping_return():
    env = _reward_only_stack(previous=None)
    env._prev_reward_potential = env._reward_potential()

    episode_return = sum(env._post_action(action=None)[0] for _ in range(201))

    assert episode_return == pytest.approx(0.0)


def test_green_misuse_penalties_are_committed_once_and_never_turn_positive():
    env = _reward_only_stack(
        snapshot={"green_grasped": True, "green_effective_shift": 0.05},
        previous=0.30,
    )
    expected = -0.30 - 0.50 - 0.125

    assert env.reward(None) == pytest.approx(expected)
    assert env.reward(None) == pytest.approx(expected)
    assert env._reward_stage is StackRewardStage.APPROACH
    assert env._prev_green_grasped is False
    assert env._prev_green_effective_shift == 0.0

    assert env._post_action(None)[0] == pytest.approx(expected)
    assert env._reward_stage is StackRewardStage.APPROACH
    assert env._prev_reward_potential == 0.0
    assert env._prev_green_grasped is True
    assert env._prev_green_effective_shift == pytest.approx(0.05)

    assert env._post_action(None)[0] == 0.0


def test_reach_and_partial_contacts_form_increasing_potential():
    far = Stack._reach_contact_potential(distance=0.30, left_contact=False, right_contact=False)
    near = Stack._reach_contact_potential(distance=0.01, left_contact=False, right_contact=False)
    single_contact = Stack._reach_contact_potential(distance=0.01, left_contact=True, right_contact=False)
    bilateral_contact = Stack._reach_contact_potential(distance=0.01, left_contact=True, right_contact=True)
    bilateral_contact_at_zero = Stack._reach_contact_potential(
        distance=0.0, left_contact=True, right_contact=True
    )

    assert 0.0 < far < near < single_contact < bilateral_contact <= 0.50
    assert far == pytest.approx(0.30 * (1.0 - np.tanh(5.0 * 0.30)))
    assert near == pytest.approx(0.30 * (1.0 - np.tanh(5.0 * 0.01)))
    assert single_contact - near == pytest.approx(0.10)
    assert bilateral_contact - single_contact == pytest.approx(0.10)
    assert bilateral_contact_at_zero == pytest.approx(0.50)


def test_grasp_contacts_checks_fingerpad_groups_independently():
    env = Stack.__new__(Stack)
    gripper = SimpleNamespace(
        important_geoms={"left_fingerpad": ["left_pad"], "right_fingerpad": ["right_pad"]}
    )
    env.robots = [SimpleNamespace(gripper=gripper)]
    env.cubeA = SimpleNamespace(contact_geoms=["cube_a"])
    env.check_contact = lambda gripper_geoms, object_geoms: (
        gripper_geoms == ["left_pad"] and object_geoms == ["cube_a"]
    )

    left_contact, right_contact = env._grasp_contacts()

    assert left_contact is True
    assert right_contact is False


def test_grasp_contacts_chooses_bilateral_contacts_from_any_arm():
    env = Stack.__new__(Stack)
    right_gripper = SimpleNamespace(
        important_geoms={"left_fingerpad": ["right_left_pad"], "right_fingerpad": ["right_right_pad"]}
    )
    left_gripper = SimpleNamespace(
        important_geoms={"left_fingerpad": ["left_left_pad"], "right_fingerpad": ["left_right_pad"]}
    )
    env.robots = [
        SimpleNamespace(
            arms=["right", "left"],
            gripper={"right": right_gripper, "left": left_gripper},
        )
    ]
    env.cubeA = SimpleNamespace(contact_geoms=["cube_a"])
    env.check_contact = lambda gripper_geoms, object_geoms: (
        gripper_geoms in (["left_left_pad"], ["left_right_pad"])
        and object_geoms == ["cube_a"]
    )

    assert env._grasp_contacts() == (True, True)
    assert env._cube_grasped() is True


def test_cube_grasped_rejects_contacts_split_across_arms():
    env = Stack.__new__(Stack)
    right_gripper = SimpleNamespace(
        important_geoms={"left_fingerpad": ["right_left_pad"], "right_fingerpad": ["right_right_pad"]}
    )
    left_gripper = SimpleNamespace(
        important_geoms={"left_fingerpad": ["left_left_pad"], "right_fingerpad": ["left_right_pad"]}
    )
    env.robots = [
        SimpleNamespace(
            arms=["right", "left"],
            gripper={"right": right_gripper, "left": left_gripper},
        )
    ]
    env.cubeA = SimpleNamespace(contact_geoms=["cube_a"])
    env.check_contact = lambda gripper_geoms, object_geoms: (
        gripper_geoms in (["right_left_pad"], ["left_right_pad"])
        and object_geoms == ["cube_a"]
    )

    left_contact, right_contact = env._grasp_contacts()

    assert int(left_contact) + int(right_contact) == 1
    assert env._cube_grasped() is False


def test_staged_rewards_includes_partial_contact_in_reach_potential():
    env = Stack.__new__(Stack)
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            body_xpos=np.array([[0.0, 0.0, 0.83], [0.20, 0.0, 0.825]]),
            site_xpos=np.array([[0.0, 0.0, 0.84]]),
        )
    )
    env.cubeA_body_id = 0
    env.cubeB_body_id = 1
    env.robots = [
        SimpleNamespace(arms=["right"], eef_site_id={"right": 0}, gripper=SimpleNamespace())
    ]
    env.cubeA = SimpleNamespace(contact_geoms=["cube_a"])
    env.cubeB = SimpleNamespace(contact_geoms=["cube_b"])
    env.table_offset = np.array([0.0, 0.0, 0.8])
    env._grasp_contacts = lambda obj=None: (True, False) if obj is None else (False, False)
    env._check_grasp = lambda *, gripper, object_geoms: False
    env.check_contact = lambda object_a, object_b: False

    r_reach, r_stage, r_stack = env.staged_rewards()

    assert r_reach == pytest.approx(
        Stack._reach_contact_potential(distance=0.01, left_contact=True, right_contact=False)
    )
    assert r_stage == pytest.approx(r_reach)
    assert r_stack == 0.0


def test_reward_stage_advances_one_physical_prerequisite_per_transition():
    lift = Stack._next_reward_stage(StackRewardStage.APPROACH, **_stage_kwargs())
    align = Stack._next_reward_stage(lift, **_stage_kwargs(red_height=0.92))
    place = Stack._next_reward_stage(
        align,
        **_stage_kwargs(red_height=0.92, horizontal_distance=0.03),
    )

    assert lift is StackRewardStage.LIFT
    assert align is StackRewardStage.ALIGN
    assert place is StackRewardStage.PLACE


def test_late_stage_cannot_be_skipped_from_coincidental_pose():
    stage = Stack._next_reward_stage(
        StackRewardStage.APPROACH,
        **_stage_kwargs(red_height=0.92, horizontal_distance=0.0, target_height_error=0.0),
    )

    assert stage is StackRewardStage.LIFT


def test_stage_potential_uses_non_overlapping_ranges():
    approach = Stack._stage_potential(
        StackRewardStage.APPROACH,
        **_stage_kwargs(red_grasped=False, red_left_contact=False, red_right_contact=False),
    )
    lift = Stack._stage_potential(StackRewardStage.LIFT, **_stage_kwargs(red_height=0.87))
    align = Stack._stage_potential(
        StackRewardStage.ALIGN,
        **_stage_kwargs(red_height=0.92, horizontal_distance=0.10),
    )
    place = Stack._stage_potential(
        StackRewardStage.PLACE,
        **_stage_kwargs(horizontal_distance=0.02, target_height_error=0.02),
    )
    success = Stack._stage_potential(
        StackRewardStage.PLACE,
        **_stage_kwargs(red_grasped=False, success=True),
    )

    assert 0.0 <= approach < 0.50
    assert 0.50 <= lift < 0.95
    assert 0.95 <= align < 1.35
    assert 1.35 <= place <= 1.70
    assert success == 2.0


def test_alignment_is_gated_by_completed_lift_and_current_transport_height():
    before_lift = Stack._stage_potential(
        StackRewardStage.LIFT,
        **_stage_kwargs(red_height=0.83, horizontal_distance=0.0),
    )
    low_after_lift = Stack._stage_potential(
        StackRewardStage.ALIGN,
        **_stage_kwargs(red_height=0.89, horizontal_distance=0.0),
    )
    high_after_lift = Stack._stage_potential(
        StackRewardStage.ALIGN,
        **_stage_kwargs(red_height=0.92, horizontal_distance=0.0),
    )
    retained_stage = Stack._next_reward_stage(
        StackRewardStage.ALIGN,
        **_stage_kwargs(red_height=0.89, horizontal_distance=0.20),
    )

    assert before_lift == pytest.approx(0.545)
    assert low_after_lift == pytest.approx(0.95)
    assert high_after_lift == pytest.approx(1.35)
    assert retained_stage is StackRewardStage.ALIGN


def test_red_drop_and_place_drift_regress_reward_stage():
    dropped = Stack._next_reward_stage(
        StackRewardStage.PLACE,
        **_stage_kwargs(red_grasped=False),
    )
    drifted = Stack._next_reward_stage(
        StackRewardStage.PLACE,
        **_stage_kwargs(horizontal_distance=0.056),
    )
    inside_hysteresis = Stack._next_reward_stage(
        StackRewardStage.PLACE,
        **_stage_kwargs(horizontal_distance=0.050),
    )

    assert dropped is StackRewardStage.APPROACH
    assert drifted is StackRewardStage.ALIGN
    assert inside_hysteresis is StackRewardStage.PLACE


def test_green_grasp_resets_stage_and_zeroes_task_potential():
    kwargs = _stage_kwargs(green_grasped=True, red_height=0.92, horizontal_distance=0.0)

    assert Stack._next_reward_stage(StackRewardStage.PLACE, **kwargs) is StackRewardStage.APPROACH
    assert Stack._stage_potential(StackRewardStage.PLACE, **kwargs) == 0.0


def test_green_grasp_and_displacement_events_are_grounded_and_nonrepeating():
    assert Stack._green_grasp_event_reward(False, True) == pytest.approx(-0.50)
    assert Stack._green_grasp_event_reward(True, True) == 0.0
    assert Stack._green_grasp_event_reward(True, False) == 0.0
    assert Stack._green_disturbance_reward(0.0, 0.05) == pytest.approx(-0.125)
    assert Stack._green_disturbance_reward(0.05, 0.05) == 0.0
    assert Stack._green_disturbance_reward(0.05, 0.0) == 0.0
    assert Stack._green_disturbance_reward(0.0, 0.20) == pytest.approx(-0.25)


def test_dense_target_geometry_is_anchored_when_green_cube_moves():
    target = np.array([0.20, 0.0, 0.825])
    red = np.array([0.20, 0.0, 0.92])
    moved_green = np.array([0.0, 0.0, 0.825])

    horizontal_distance, target_height_error = Stack._target_geometry(red, target)

    assert horizontal_distance == 0.0
    assert target_height_error == pytest.approx(abs(0.92 - (0.825 + 0.045)))
    assert Stack._green_effective_shift(moved_green, target) == pytest.approx(0.19)


def test_moving_green_under_fixed_red_cannot_improve_active_stage_potential():
    env = Stack.__new__(Stack)
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            body_xpos=np.array([[0.0, 0.0, 0.92], [0.20, 0.0, 0.825]]),
            site_xpos=np.array([[0.0, 0.0, 0.92]]),
        )
    )
    env.cubeA_body_id = 0
    env.cubeB_body_id = 1
    env.robots = [
        SimpleNamespace(arms=["right"], eef_site_id={"right": 0}, gripper=SimpleNamespace())
    ]
    env.cubeA = SimpleNamespace(contact_geoms=["cube_a"])
    env.cubeB = SimpleNamespace(contact_geoms=["cube_b"])
    env.table_offset = np.array([0.0, 0.0, 0.8])
    env._stack_reward_target_pos = np.array([0.20, 0.0, 0.825])
    env._grasp_contacts = lambda obj=None: (True, True) if obj is None else (False, False)
    env.check_contact = lambda object_a, object_b: False

    before = env._reward_snapshot()
    before_potential = Stack._stage_potential(StackRewardStage.ALIGN, **before)
    env.sim.data.body_xpos[env.cubeB_body_id] = np.array([0.0, 0.0, 0.825])
    after = env._reward_snapshot()
    after_potential = Stack._stage_potential(StackRewardStage.ALIGN, **after)

    assert before["horizontal_distance"] == pytest.approx(0.20)
    assert after["horizontal_distance"] == pytest.approx(0.20)
    assert after_potential == pytest.approx(before_potential)
    assert after["green_effective_shift"] == pytest.approx(0.19)
    assert Stack._green_disturbance_reward(0.0, after["green_effective_shift"]) < 0.0


def test_reset_seeds_pbrs_from_the_fully_forwarded_initial_state():
    env = suite.make(
        env_name="Stack",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        hard_reset=False,
    )
    try:
        env.reset()

        assert env._reward_stage is StackRewardStage.APPROACH
        assert env._stack_reward_target_pos == pytest.approx(
            env.sim.data.body_xpos[env.cubeB_body_id]
        )
        assert np.isfinite(env._prev_reward_potential)
        assert env._prev_reward_potential == pytest.approx(env._reward_potential())
        assert env._prev_red_grasped == env._cube_grasped(env.cubeA)
        assert env._prev_green_grasped == env._cube_grasped(env.cubeB)
        assert env._prev_green_effective_shift == 0.0
    finally:
        env.close()


def test_sparse_reset_clears_reward_shaping_caches():
    env = suite.make(
        env_name="Stack",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=False,
        hard_reset=False,
    )
    try:
        env.reset()

        assert env._reward_stage is StackRewardStage.APPROACH
        assert env._stack_reward_target_pos == pytest.approx(
            env.sim.data.body_xpos[env.cubeB_body_id]
        )
        assert env._prev_reward_potential is None
        assert env._prev_red_grasped is None
        assert env._prev_green_grasped is None
        assert env._prev_green_effective_shift is None
    finally:
        env.close()
