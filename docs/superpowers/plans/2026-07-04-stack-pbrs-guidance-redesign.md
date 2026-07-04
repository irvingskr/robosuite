# Stack PBRS Guidance Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Stack a stronger, physically grounded reach-to-stack learning signal while retaining `0.99 * Phi(s') - Phi(s)` and preventing positive no-progress rewards.

**Architecture:** Keep `staged_rewards()` and its maximum as the state potential, but make the reach/contact and transport stages cumulative and continuous. Add paired grasp-acquisition and invalid-drop transition terms in `_compute_reward()`, with all reward history updated only by `_post_action()` and initialized after reset pose propagation.

**Tech Stack:** Python 3.10, NumPy, MuJoCo through robosuite, pytest, HIRL Gymnasium wrappers

---

## File Structure

- Modify `robosuite/environments/manipulation/stack.py`: potential components, finger-contact detection, event reward, and reward-state lifecycle.
- Modify `tests/test_environments/test_stack.py`: pure geometry, event arithmetic, state update, reset, and compatibility regressions.
- Modify `CHANGELOG.md`: document the denser PBRS stages and paired physical events.
- Reference `docs/superpowers/specs/2026-07-04-stack-pbrs-guidance-redesign.md`: approved behavior and empirical thresholds.

### Task 1: Reach and partial-contact potential

**Files:**
- Modify: `tests/test_environments/test_stack.py`
- Modify: `robosuite/environments/manipulation/stack.py:14`

- [ ] **Step 1: Write the failing reach/contact test**

Add to `tests/test_environments/test_stack.py`:

```python
def test_reach_and_partial_contacts_form_increasing_potential():
    far = Stack._reach_contact_potential(
        distance=0.30,
        left_contact=False,
        right_contact=False,
    )
    near = Stack._reach_contact_potential(
        distance=0.01,
        left_contact=False,
        right_contact=False,
    )
    single_contact = Stack._reach_contact_potential(
        distance=0.01,
        left_contact=True,
        right_contact=False,
    )
    bilateral_contact = Stack._reach_contact_potential(
        distance=0.01,
        left_contact=True,
        right_contact=True,
    )

    assert 0.0 < far < near < single_contact < bilateral_contact <= 0.60
    assert single_contact - near == pytest.approx(0.125)
    assert bilateral_contact - single_contact == pytest.approx(0.125)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py::test_reach_and_partial_contacts_form_increasing_potential -q
```

Expected: FAIL with `AttributeError: type object 'Stack' has no attribute '_reach_contact_potential'`.

- [ ] **Step 3: Implement the reach/contact potential**

Add alongside the existing Stack reward constants in
`robosuite/environments/manipulation/stack.py`:

```python
STACK_REACH_DISTANCE_SCALE = 5.0
STACK_REACH_WEIGHT = 0.35
STACK_CONTACT_WEIGHT = 0.25
```

Add before `_lift_align_place_potential()`:

```python
@staticmethod
def _reach_contact_potential(distance, left_contact, right_contact):
    reach = 1.0 - np.tanh(STACK_REACH_DISTANCE_SCALE * distance)
    contact = 0.5 * (float(left_contact) + float(right_contact))
    return float(STACK_REACH_WEIGHT * reach + STACK_CONTACT_WEIGHT * contact)
```

- [ ] **Step 4: Run the test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Commit the focused change**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "test: define Stack reach contact potential"
```

### Task 2: Cumulative lift, alignment, and placement potential

**Files:**
- Modify: `tests/test_environments/test_stack.py`
- Modify: `robosuite/environments/manipulation/stack.py:18`
- Modify: `robosuite/environments/manipulation/stack.py:291`

- [ ] **Step 1: Replace the geometry tests with the approved sequence**

Replace `test_lift_align_place_potential_increases_through_successful_motion`
with:

```python
def test_lift_align_place_potential_increases_through_successful_motion():
    table_height = 0.8
    cube_b = np.array([0.20, 0.0, 0.825])
    target_height = cube_b[2] + 0.045
    base_potential = 0.60

    grasped = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.0, 0.0, table_height + 0.03]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
        base_potential=base_potential,
    )
    lifted = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.0, 0.0, table_height + 0.18]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
        base_potential=base_potential,
    )
    aligned = Stack._lift_align_place_potential(
        cube_a_pos=np.array([cube_b[0], cube_b[1], target_height + 0.10]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
        base_potential=base_potential,
    )
    placed = Stack._lift_align_place_potential(
        cube_a_pos=np.array([cube_b[0], cube_b[1], target_height]),
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
        base_potential=base_potential,
    )

    assert base_potential == pytest.approx(grasped)
    assert grasped < lifted < aligned < placed
    assert placed == pytest.approx(1.80)
```

Add:

```python
def test_placement_uses_absolute_target_height_error():
    table_height = 0.8
    cube_b = np.array([0.20, 0.0, 0.825])
    target_height = cube_b[2] + 0.045
    kwargs = dict(
        cube_b_pos=cube_b,
        table_height=table_height,
        grasping=True,
        base_potential=0.60,
    )

    target = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.20, 0.0, target_height]),
        **kwargs,
    )
    above = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.20, 0.0, target_height + 0.02]),
        **kwargs,
    )
    below = Stack._lift_align_place_potential(
        cube_a_pos=np.array([0.20, 0.0, target_height - 0.02]),
        **kwargs,
    )

    assert above == pytest.approx(below)
    assert above < target
```

Pass `base_potential=0.0` in
`test_lift_align_place_potential_is_zero_for_untouched_table_cube`.

- [ ] **Step 2: Run both geometry tests and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py::test_lift_align_place_potential_increases_through_successful_motion \
  tests/test_environments/test_stack.py::test_placement_uses_absolute_target_height_error -q
```

Expected: FAIL because `_lift_align_place_potential()` does not accept
`base_potential`, and the current placement term is not symmetric.

- [ ] **Step 3: Implement cumulative transport geometry**

Replace the old geometry constants with:

```python
STACK_LIFT_START_HEIGHT = 0.03
STACK_LIFT_PROGRESS_HEIGHT = 0.12
STACK_ALIGNMENT_DISTANCE_SCALE = 10.0
STACK_PLACEMENT_HEIGHT = 0.10
STACK_LIFT_WEIGHT = 0.55
STACK_ALIGNMENT_WEIGHT = 0.40
STACK_PLACEMENT_WEIGHT = 0.25
STACK_PRE_SUCCESS_POTENTIAL_MAX = 1.80
```

Replace `_lift_align_place_potential()` with:

```python
@staticmethod
def _lift_align_place_potential(
    cube_a_pos,
    cube_b_pos,
    table_height,
    grasping,
    base_potential,
):
    lift = np.clip(
        (cube_a_pos[2] - (table_height + STACK_LIFT_START_HEIGHT))
        / STACK_LIFT_PROGRESS_HEIGHT,
        0.0,
        1.0,
    )
    cube_lifted = cube_a_pos[2] > table_height + 0.04
    alignment = 0.0
    placement = 0.0
    if cube_lifted:
        horizontal_distance = np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2])
        alignment = 1.0 - np.tanh(
            STACK_ALIGNMENT_DISTANCE_SCALE * horizontal_distance
        )
        target_height = cube_b_pos[2] + 0.045
        height_error = abs(cube_a_pos[2] - target_height)
        placement = 1.0 - np.clip(
            height_error / STACK_PLACEMENT_HEIGHT,
            0.0,
            1.0,
        )

    if not grasping and not cube_lifted:
        return 0.0

    potential = (
        base_potential
        + STACK_LIFT_WEIGHT * max(lift, alignment)
        + STACK_ALIGNMENT_WEIGHT * alignment
        + STACK_PLACEMENT_WEIGHT * alignment**2 * placement
    )
    return float(min(potential, STACK_PRE_SUCCESS_POTENTIAL_MAX))
```

- [ ] **Step 4: Run the geometry tests and verify GREEN**

Run the command from Step 2 plus the untouched-cube test.

Expected: `3 passed`.

- [ ] **Step 5: Commit the focused change**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "feat: make Stack transport potential cumulative"
```

### Task 3: Finger contacts in staged rewards

**Files:**
- Modify: `tests/test_environments/test_stack.py`
- Modify: `robosuite/environments/manipulation/stack.py:314`

- [ ] **Step 1: Write failing contact-group tests**

Add `SimpleNamespace` to the test imports:

```python
from types import SimpleNamespace
```

Add:

```python
def test_grasp_contacts_checks_fingerpad_groups_independently():
    env = Stack.__new__(Stack)
    gripper = SimpleNamespace(
        important_geoms={
            "left_fingerpad": ["left_pad"],
            "right_fingerpad": ["right_pad"],
        }
    )
    env.robots = [SimpleNamespace(gripper=gripper)]
    env.cubeA = SimpleNamespace(contact_geoms=["cube_a"])
    env.check_contact = lambda gripper_geoms, object_geoms: (
        gripper_geoms == ["left_pad"] and object_geoms == ["cube_a"]
    )

    left_contact, right_contact = env._grasp_contacts()

    assert left_contact is True
    assert right_contact is False
```

- [ ] **Step 2: Run the contact test and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py::test_grasp_contacts_checks_fingerpad_groups_independently -q
```

Expected: FAIL with `AttributeError: 'Stack' object has no attribute '_grasp_contacts'`.

- [ ] **Step 3: Implement contact detection and integrate staged rewards**

Add before `staged_rewards()`:

```python
def _grasp_contacts(self):
    gripper = self.robots[0].gripper
    object_geoms = self.cubeA.contact_geoms
    left_contact = self.check_contact(
        gripper.important_geoms["left_fingerpad"],
        object_geoms,
    )
    right_contact = self.check_contact(
        gripper.important_geoms["right_fingerpad"],
        object_geoms,
    )
    return bool(left_contact), bool(right_contact)

def _cube_grasped(self):
    left_contact, right_contact = self._grasp_contacts()
    return left_contact and right_contact
```

In `staged_rewards()`, replace the old reach and grasp calculation with:

```python
left_contact, right_contact = self._grasp_contacts()
grasping_cubeA = left_contact and right_contact
r_reach = self._reach_contact_potential(
    distance=dist,
    left_contact=left_contact,
    right_contact=right_contact,
)
```

Pass `base_potential=r_reach` to `_lift_align_place_potential()`. Preserve the
existing strict stack condition using `grasping_cubeA`, `cubeA_lifted`, and
cube-to-cube contact.

- [ ] **Step 4: Run contact, geometry, and reset tests**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -q
```

Expected: the new contact and geometry tests pass; event tests are not present
yet, and all existing PBRS tests remain green.

- [ ] **Step 5: Commit the focused change**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "feat: add partial contacts to Stack potential"
```

### Task 4: Grasp acquisition and invalid-drop events

**Files:**
- Modify: `tests/test_environments/test_stack.py:8`
- Modify: `robosuite/environments/manipulation/stack.py:14`
- Modify: `robosuite/environments/manipulation/stack.py:271`

- [ ] **Step 1: Extend the reward-only fixture**

Replace `_reward_only_stack()` with:

```python
def _reward_only_stack(
    *,
    stages,
    previous,
    reward_shaping=True,
    reward_scale=None,
    grasped=False,
    previous_grasped=False,
):
    env = Stack.__new__(Stack)
    env.reward_shaping = reward_shaping
    env.reward_scale = reward_scale
    env._prev_reward_potential = previous
    env._prev_grasped = previous_grasped
    env.staged_rewards = lambda: stages
    env._cube_grasped = lambda: grasped
    env.timestep = 1
    env.horizon = 100
    env.ignore_done = False
    env.done = False
    return env
```

- [ ] **Step 2: Write failing event and idempotence tests**

Add:

```python
def test_grasp_acquisition_bonus_is_one_time_and_read_only_calls_are_idempotent():
    env = _reward_only_stack(
        stages=(0.60, 0.60, 0.0),
        previous=0.35,
        grasped=True,
        previous_grasped=False,
    )
    expected = 0.99 * 0.60 - 0.35 + 0.25

    assert env.reward(action=None) == pytest.approx(expected)
    assert env.reward(action=None) == pytest.approx(expected)
    assert env._prev_reward_potential == pytest.approx(0.35)
    assert env._prev_grasped is False


def test_invalid_drop_is_penalized_but_successful_release_is_not():
    dropped = _reward_only_stack(
        stages=(0.35, 0.35, 0.0),
        previous=0.60,
        grasped=False,
        previous_grasped=True,
    )
    successful_release = _reward_only_stack(
        stages=(0.35, 1.80, 2.0),
        previous=1.80,
        grasped=False,
        previous_grasped=True,
    )

    assert dropped.reward(action=None) == pytest.approx(
        0.99 * 0.35 - 0.60 - 0.35
    )
    assert successful_release.reward(action=None) == pytest.approx(
        2.0 + 0.99 * 2.0 - 1.80
    )


def test_complete_grasp_drop_event_pair_is_negative():
    acquire = Stack._grasp_event_reward(
        previous_grasped=False,
        grasped=True,
        success=False,
    )
    drop = Stack._grasp_event_reward(
        previous_grasped=True,
        grasped=False,
        success=False,
    )

    assert acquire > 0.0
    assert drop < 0.0
    assert acquire + drop < 0.0
```

Update `test_post_action_updates_potential_once_and_stalling_has_discount_cost`
to assert `env._prev_grasped is False` after both calls.

- [ ] **Step 3: Run the event tests and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py::test_grasp_acquisition_bonus_is_one_time_and_read_only_calls_are_idempotent \
  tests/test_environments/test_stack.py::test_invalid_drop_is_penalized_but_successful_release_is_not \
  tests/test_environments/test_stack.py::test_complete_grasp_drop_event_pair_is_negative -q
```

Expected: FAIL because event reward is not implemented.

- [ ] **Step 4: Implement paired event reward and once-per-step updates**

Add constants:

```python
STACK_GRASP_ACQUIRED_REWARD = 0.25
STACK_GRASP_LOST_PENALTY = 0.35
```

Add:

```python
@staticmethod
def _grasp_event_reward(previous_grasped, grasped, success):
    if grasped and not previous_grasped:
        return STACK_GRASP_ACQUIRED_REWARD
    if previous_grasped and not grasped and not success:
        return -STACK_GRASP_LOST_PENALTY
    return 0.0
```

Replace `_compute_reward()` with:

```python
def _compute_reward(self, action=None, update_reward_state=False):
    success = self._check_success()
    sparse_reward = STACK_SUCCESS_REWARD if success else 0.0
    reward = sparse_reward
    if self.reward_shaping:
        potential = self._reward_potential()
        grasped = self._cube_grasped()
        prev_potential = getattr(self, "_prev_reward_potential", None)
        prev_grasped = getattr(self, "_prev_grasped", None)
        if prev_potential is not None:
            reward += REWARD_SHAPING_GAMMA * potential - prev_potential
        if prev_grasped is not None:
            reward += self._grasp_event_reward(
                previous_grasped=prev_grasped,
                grasped=grasped,
                success=success,
            )
        if update_reward_state:
            self._prev_reward_potential = potential
            self._prev_grasped = grasped

    if self.reward_scale is not None:
        reward *= self.reward_scale / STACK_SUCCESS_REWARD
    return reward
```

- [ ] **Step 5: Run all Stack tests and verify GREEN**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -q
```

Expected: all Stack tests pass.

- [ ] **Step 6: Commit the focused change**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "feat: reward Stack grasp and penalize drops"
```

### Task 5: Reset lifecycle and documentation

**Files:**
- Modify: `tests/test_environments/test_stack.py:129`
- Modify: `robosuite/environments/manipulation/stack.py:228`
- Modify: `robosuite/environments/manipulation/stack.py:456`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Strengthen the reset regression test**

Add this assertion to
`test_reset_seeds_pbrs_from_the_fully_forwarded_initial_state`:

```python
assert env._prev_grasped == env._cube_grasped()
```

- [ ] **Step 2: Run the reset test and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py::test_reset_seeds_pbrs_from_the_fully_forwarded_initial_state -q
```

Expected: FAIL because `_prev_grasped` is not initialized.

- [ ] **Step 3: Seed both shaping caches after reset**

Replace the shaping-state block in `_reset_internal()` with:

```python
if self.reward_shaping:
    self.sim.forward()
    self._prev_reward_potential = self._reward_potential()
    self._prev_grasped = self._cube_grasped()
else:
    self._prev_reward_potential = None
    self._prev_grasped = None
```

- [ ] **Step 4: Update reward documentation**

In `Stack.reward()`'s docstring, describe the `0.35` reach, `0.25` partial
contact, `1.8` cumulative transport, `2.0` success potential, one-time `0.25`
grasp event, and `-0.35` invalid-drop event. State explicitly that repeated
holding has no bonus and remains subject to the PBRS discount cost.

Add under `CHANGELOG.md`'s Unreleased Features:

```markdown
- Strengthened Stack PBRS guidance with partial finger contacts, cumulative transport progress, a one-time grasp reward, and an invalid-drop penalty.
```

Add under Design Rationale:

```markdown
- Kept Stack's no-progress PBRS reward non-positive while pairing grasp acquisition with a larger pre-success grasp-loss penalty to prevent reward cycling.
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -q
```

Expected: all Stack tests pass.

- [ ] **Step 6: Commit the lifecycle and documentation change**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py CHANGELOG.md
git commit -m "docs: explain guided Stack PBRS reward"
```

### Task 6: Regression and HIRL trajectory verification

**Files:**
- Verify: `robosuite/environments/manipulation/stack.py`
- Verify: `tests/test_environments/test_stack.py`
- Verify: `/data/ChenZihan/project/HIRL/robotenv/stack_auto_collect_wrapper.py`

- [ ] **Step 1: Run focused environment regressions**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py \
  tests/test_environments/test_all_environments.py -q
```

Expected: all selected tests pass with no warnings introduced by Stack.

- [ ] **Step 2: Run one successful HIRL auto-controller batch without saving data**

Run from `/data/ChenZihan/project/HIRL`:

```bash
PYTHONPATH=/data/ChenZihan/project/HIRL \
/data/ChenZihan/miniforge3/envs/hilserl/bin/python - <<'PY'
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from robotenv.env_config import create_arx_env
from robotenv.stack_auto_collect_wrapper import _find_stack_env

config_dir = str((Path.cwd() / "config").resolve())
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(
        config_name="config_serl",
        overrides=[
            "env.sim.reward_shaping=true",
            "env.sim.has_renderer=false",
            "env.wrappers.intervention_mode=stack_auto_collect",
            "env.wrappers.step_limit.max_step=200",
        ],
    )

env = create_arx_env(
    cfg,
    use_classifier=False,
    use_chunking=False,
    use_human_intervention=True,
    use_step_limit=True,
    use_gripper_close=False,
    has_renderer=False,
)

try:
    acquisition_rewards = []
    approach_positive_rewards = []
    preterminal_returns = []
    for episode in range(10):
        _, _ = env.reset(seed=cfg.common.seed + episode)
        task = _find_stack_env(env)
        previous_grasped = task._cube_grasped()
        rewards = []
        env.get_wrapper_attr("start_intervention")()
        while True:
            _, reward, done, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            rewards.append(float(reward))
            grasped = task._cube_grasped()
            if grasped and not previous_grasped:
                acquisition_rewards.append(float(reward))
            previous_grasped = grasped
            if info.get("current_phase") == "approach" and reward > 0.0:
                approach_positive_rewards.append(float(reward))
            if done or truncated:
                if not info.get("success", False):
                    raise AssertionError(f"episode {episode} did not stack")
                preterminal_returns.append(float(np.sum(rewards[:-1])))
                break

    print("acquisition rewards:", acquisition_rewards)
    print("median positive approach reward:", np.median(approach_positive_rewards))
    print("mean preterminal return:", np.mean(preterminal_returns))
    assert len(acquisition_rewards) >= 10
    assert min(acquisition_rewards) > 0.0
    assert np.median(approach_positive_rewards) > 0.000166
    assert np.mean(preterminal_returns) > 0.4854
finally:
    env.close()
PY
```

Expected: ten successful episodes; every grasp-acquisition reward is positive;
the median positive approach reward exceeds `0.000166`; mean preterminal return
exceeds `0.4854`.

- [ ] **Step 3: Run repository hygiene checks**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` prints nothing. Status contains only intentional
files if the task commits have not already made the worktree clean.

- [ ] **Step 4: Inspect the final diff against scope**

Run:

```bash
git diff e2dd210..HEAD -- \
  robosuite/environments/manipulation/stack.py \
  tests/test_environments/test_stack.py \
  CHANGELOG.md
```

Expected: changes are limited to Stack reward behavior, its tests, and the
changelog; object placement, observations, sparse success, and PegInsertion are
unchanged.
