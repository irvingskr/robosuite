import numpy as np
import pytest

import robosuite as suite
from robosuite.environments.manipulation.stack import Stack


def _reward_only_stack(*, stages, previous, reward_shaping=True, reward_scale=None):
    env = Stack.__new__(Stack)
    env.reward_shaping = reward_shaping
    env.reward_scale = reward_scale
    env._prev_reward_potential = previous
    env.staged_rewards = lambda: stages
    env.timestep = 1
    env.horizon = 100
    env.ignore_done = False
    env.done = False
    return env


def test_shaped_reward_adds_sparse_reward_and_discounted_potential_difference():
    env = _reward_only_stack(stages=(0.2, 0.7, 2.0), previous=0.4)

    reward = env.reward(action=None)

    assert reward == pytest.approx(2.0 + 0.99 * 2.0 - 0.4)
    assert env._prev_reward_potential == 0.4


def test_reward_potential_is_zero_without_progress_and_positive_at_success():
    env = _reward_only_stack(stages=(0.0, 0.0, 0.0), previous=0.0)

    assert env._reward_potential() == 0.0

    env.staged_rewards = lambda: (0.2, 0.7, 2.0)
    assert env._reward_potential() == pytest.approx(2.0)


def test_sparse_reward_and_existing_scaling_are_preserved_without_shaping():
    env = _reward_only_stack(
        stages=(0.2, 0.7, 2.0),
        previous=0.4,
        reward_shaping=False,
        reward_scale=3.0,
    )

    assert env.reward(action=None) == pytest.approx(3.0)
    assert env._prev_reward_potential == 0.4


def test_post_action_updates_potential_once_and_stalling_has_discount_cost():
    stages = [0.2, 0.7, 0.0]
    env = _reward_only_stack(stages=stages, previous=0.4)

    first_reward, first_done, first_info = env._post_action(action=None)
    second_reward, second_done, second_info = env._post_action(action=None)

    assert first_reward == pytest.approx(0.99 * 0.7 - 0.4)
    assert second_reward == pytest.approx(0.99 * 0.7 - 0.7)
    assert first_reward > 0.0
    assert second_reward < 0.0
    assert env._prev_reward_potential == pytest.approx(0.7)
    assert first_done is False
    assert second_done is False
    assert first_info == {}
    assert second_info == {}


def test_no_progress_rollout_does_not_accumulate_positive_shaping_return():
    env = _reward_only_stack(stages=(0.0, 0.0, 0.0), previous=None)
    env._prev_reward_potential = env._reward_potential()

    episode_return = sum(env._post_action(action=None)[0] for _ in range(201))

    assert episode_return == pytest.approx(0.0)


def test_lift_align_place_potential_increases_through_successful_motion():
    table_height = 0.8
    cube_b = np.array([0.20, 0.0, 0.825])
    target_height = cube_b[2] + 0.045

    grasped = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.0, 0.0, table_height + 0.03]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
    )
    lifted = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.0, 0.0, table_height + 0.18]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
    )
    aligned = Stack._lift_align_place_potential(
        cube_a_pos=np.array([cube_b[0], cube_b[1], target_height + 0.10]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
    )
    placed = Stack._lift_align_place_potential(
        cube_a_pos=np.array([cube_b[0], cube_b[1], target_height]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
    )

    assert 0.0 < grasped < lifted < aligned < placed < 2.0


def test_lift_align_place_potential_is_zero_for_untouched_table_cube():
    resting = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.0, 0.0, 0.83]),
        cube_b_pos=np.array([0.20, 0.0, 0.825]),
        table_height=0.8,
        grasping=False,
    )
    contact_jitter = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.0, 0.0, 0.839]),
        cube_b_pos=np.array([0.20, 0.0, 0.825]),
        table_height=0.8,
        grasping=False,
    )

    assert resting == 0.0
    assert contact_jitter == 0.0


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

        assert np.isfinite(env._prev_reward_potential)
        assert env._prev_reward_potential == pytest.approx(env._reward_potential())
    finally:
        env.close()
