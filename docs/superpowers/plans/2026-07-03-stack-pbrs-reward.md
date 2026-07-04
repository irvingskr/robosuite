# Stack PBRS Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Stack dense reward shaping from repeated absolute phase scores to `r_sparse + 0.99 * Phi(s') - Phi(s)` while preserving sparse reward scaling and existing staged rewards.

**Architecture:** Keep `staged_rewards()` as the single definition of Stack progress and define `Phi` as its maximum component. Mirror PegInsertion's read-only `_compute_reward()` / state-updating `_post_action()` split, then seed the previous potential from the fully forwarded reset state.

**Tech Stack:** Python, NumPy, robosuite / MuJoCo, pytest

---

### Task 1: PBRS arithmetic and once-per-step state updates

**Files:**
- Create: `tests/test_environments/test_stack.py`
- Modify: `robosuite/environments/manipulation/stack.py:12,219-294`

- [x] **Step 1: Write failing reward tests**

Create `tests/test_environments/test_stack.py` with reward-only Stack instances so the arithmetic is tested independently of MuJoCo contacts:

```python
import pytest

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
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest tests/test_environments/test_stack.py -q
```

Expected: the first and third tests fail because current Stack returns an absolute maximum and inherited `_post_action()` does not update `_prev_reward_potential`.

- [x] **Step 3: Implement the minimal PBRS computation**

Add the module constant after imports:

```python
REWARD_SHAPING_GAMMA = 0.99
```

Replace `reward()` and add the helper methods below it:

```python
    def _reward_potential(self):
        return float(max(self.staged_rewards()))

    def _compute_reward(self, action=None, update_reward_state=False):
        sparse_reward = 2.0 if self._check_success() else 0.0
        reward = sparse_reward
        if self.reward_shaping:
            potential = self._reward_potential()
            prev_potential = getattr(self, "_prev_reward_potential", None)
            if prev_potential is not None:
                reward += REWARD_SHAPING_GAMMA * potential - prev_potential
            if update_reward_state:
                self._prev_reward_potential = potential

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0
        return reward

    def reward(self, action=None):
        return self._compute_reward(action=action, update_reward_state=False)

    def _post_action(self, action):
        reward = self._compute_reward(action=action, update_reward_state=True)
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, {}
```

Update the reward docstring to state that shaped reward is the sparse reward plus `0.99 * Phi(s') - Phi(s)`, where `Phi` is the maximum staged score. Remove the old claim that shaped reward directly returns that maximum.

- [x] **Step 4: Run the tests and verify GREEN**

Run:

```bash
pytest tests/test_environments/test_stack.py -q
```

Expected: `3 passed`.

### Task 2: Initial-potential caching after reset

**Files:**
- Modify: `tests/test_environments/test_stack.py`
- Modify: `robosuite/environments/manipulation/stack.py:423-439`

- [x] **Step 1: Write the failing reset test**

Append this integration test and imports:

```python
import numpy as np

import robosuite as suite


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
```

- [x] **Step 2: Run the reset test and verify RED**

Run:

```bash
pytest tests/test_environments/test_stack.py::test_reset_seeds_pbrs_from_the_fully_forwarded_initial_state -q
```

Expected: FAIL because the current reset path does not seed `_prev_reward_potential`.

- [x] **Step 3: Seed the potential at the end of `_reset_internal()`**

After object placement is complete, add:

```python
        if self.reward_shaping:
            self.sim.forward()
            self._prev_reward_potential = self._reward_potential()
        else:
            self._prev_reward_potential = None
```

The explicit forward is required because object joint poses were just written and body / contact state must be current before computing `Phi(s0)`.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest tests/test_environments/test_stack.py -q
```

Expected: `4 passed`.

### Task 3: Regression verification and clean handoff

**Files:**
- Verify: `robosuite/environments/manipulation/stack.py`
- Verify: `tests/test_environments/test_stack.py`
- Verify: `robosuite/environments/manipulation/peg_insertion.py`

- [x] **Step 1: Run reward and neighboring environment tests**

Run:

```bash
pytest tests/test_environments/test_stack.py tests/test_environments/test_peg_insertion.py -q
```

Expected: `27 passed`; the known JAX CUDA plugin compatibility warning may remain and is unrelated.

- [x] **Step 2: Run formatting and syntax checks**

Run:

```bash
python -m py_compile robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
black --check robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git diff --check
```

Expected: every command exits with status 0.

- [x] **Step 3: Inspect the final diff and preserve user changes**

Run:

```bash
git diff -- robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py robosuite/environments/manipulation/peg_insertion.py
```

Confirm that Stack object placement remains `[0.15, 0.30] / [-0.15, 0.15]`, PegInsertion placement remains untouched by this implementation, and all new Stack edits are limited to reward behavior and its tests.

- [x] **Step 4: Do not stage or commit the dirty production files automatically**

The checkout was already dirty in both environment files before implementation. Leave the verified implementation in the working tree so the user's placement edits are neither accidentally staged nor folded into an unrelated automated commit.
