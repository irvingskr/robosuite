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
    assert env._prev_reward_potential == 0.7
    assert first_done is False
    assert second_done is False
    assert first_info == {}
    assert second_info == {}


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
