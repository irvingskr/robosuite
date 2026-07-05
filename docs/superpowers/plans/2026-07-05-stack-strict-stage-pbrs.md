# Stack Strict-Stage PBRS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Stack's unordered transport potential with a strict, regressible red-cube task sequence and prevent cube-B manipulation from earning dense reward.

**Architecture:** Keep sparse success and `0.99 * Phi(s') - Phi(s)`, but define `Phi` over simulator state plus an internal `StackRewardStage`. Pure helpers own stage transitions, stage-bounded potentials, and cube-B disturbance arithmetic; `_compute_reward()` derives a candidate transition read-only, while `_post_action()` commits it once.

**Tech Stack:** Python 3.10, NumPy, MuJoCo through robosuite, pytest, HIRL Stack auto-controller

---

## File Structure

- Modify `robosuite/environments/manipulation/stack.py`: stage enum, stage transition and potential helpers, generalized cube contacts, transition penalties, and cache lifecycle.
- Modify `tests/test_environments/test_stack.py`: ordered-stage, anti-cube-B, reward arithmetic, reset, sparse compatibility, and idempotence tests.
- Modify `CHANGELOG.md`: document strict ordered shaping and dense-data incompatibility.
- Reference `docs/superpowers/specs/2026-07-05-stack-strict-stage-pbrs-design.md`: approved thresholds and invariants.

### Task 1: Specify ordered stages and bounded potentials

**Files:**
- Modify: `tests/test_environments/test_stack.py`
- Modify: `robosuite/environments/manipulation/stack.py:1-30`

- [ ] **Step 1: Replace the old cumulative-transport tests with failing strict-stage tests**

Import `StackRewardStage`, then add tests that call the wished-for pure API:

```python
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


def test_reward_stage_advances_one_physical_prerequisite_per_transition():
    lift = Stack._next_reward_stage(StackRewardStage.APPROACH, **_stage_kwargs())
    align = Stack._next_reward_stage(
        lift,
        **_stage_kwargs(red_height=0.92),
    )
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

    assert before_lift == pytest.approx(0.545)
    assert low_after_lift == pytest.approx(0.95)
    assert high_after_lift == pytest.approx(1.35)
```

Remove tests for `_lift_align_place_potential()`, because that unordered helper
is intentionally deleted.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -k 'reward_stage or late_stage or stage_potential or alignment_is_gated' -q
```

Expected: collection fails because `StackRewardStage` and the two helpers do not
exist.

- [ ] **Step 3: Define stage constants, transition rules, and pure potential helpers**

Add `IntEnum`, replace the old transport weights with the approved constants,
and implement:

```python
class StackRewardStage(IntEnum):
    APPROACH = 0
    LIFT = 1
    ALIGN = 2
    PLACE = 3


@staticmethod
def _next_reward_stage(previous_stage, *, red_grasped, green_grasped,
                       red_height, table_height, horizontal_distance,
                       success, **_):
    previous_stage = StackRewardStage(previous_stage)
    if success:
        return previous_stage
    if green_grasped or not red_grasped:
        return StackRewardStage.APPROACH
    if previous_stage is StackRewardStage.APPROACH:
        return StackRewardStage.LIFT
    if previous_stage is StackRewardStage.LIFT and red_height >= table_height + 0.12:
        return StackRewardStage.ALIGN
    if (
        previous_stage is StackRewardStage.ALIGN
        and red_height >= table_height + 0.10
        and horizontal_distance <= 0.035
    ):
        return StackRewardStage.PLACE
    if previous_stage is StackRewardStage.PLACE and horizontal_distance > 0.055:
        return StackRewardStage.ALIGN
    return previous_stage


@staticmethod
def _stage_potential(stage, *, distance, red_left_contact,
                     red_right_contact, red_grasped, green_grasped,
                     red_height, table_height, horizontal_distance,
                     target_height_error, success):
    if success:
        return STACK_SUCCESS_REWARD
    if green_grasped:
        return 0.0
    stage = StackRewardStage(stage)
    if stage is StackRewardStage.APPROACH:
        reach = 1.0 - np.tanh(5.0 * distance)
        return float(
            0.30 * reach
            + 0.10 * float(red_left_contact)
            + 0.10 * float(red_right_contact)
        )
    if not red_grasped:
        return 0.0
    if stage is StackRewardStage.LIFT:
        lift = np.clip((red_height - (table_height + 0.02)) / 0.10, 0.0, 1.0)
        return float(0.50 + 0.45 * lift)
    if stage is StackRewardStage.ALIGN:
        if red_height < table_height + 0.10:
            return 0.95
        alignment = 1.0 - np.tanh(10.0 * horizontal_distance)
        return float(0.95 + 0.40 * alignment)
    placement = 1.0 - np.clip(target_height_error / 0.10, 0.0, 1.0)
    return float(1.35 + 0.35 * placement)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Commit the focused stage model**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "test: specify ordered Stack reward stages"
```

### Task 2: Specify cube-B misuse and regression behavior

**Files:**
- Modify: `tests/test_environments/test_stack.py`
- Modify: `robosuite/environments/manipulation/stack.py:300-390`

- [ ] **Step 1: Add failing regression and cube-B tests**

```python
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


def test_dense_target_is_reset_anchored_when_green_cube_moves():
    target = np.array([0.20, 0.0, 0.825])
    red = np.array([0.20, 0.0, 0.92])
    fixed = Stack._target_geometry(red, target)
    moved = Stack._target_geometry(red, target)

    assert moved == fixed
```

The last test intentionally does not pass live cube-B position to
`_target_geometry`; this makes the reset anchor part of the helper contract.

- [ ] **Step 2: Run these tests and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -k 'green or regress or reset_anchored' -q
```

Expected: failures for missing cube-B event and geometry helpers.

- [ ] **Step 3: Add grounded cube-B helpers and generalized contacts**

Implement:

```python
@staticmethod
def _target_geometry(cube_a_pos, target_pos):
    horizontal_distance = np.linalg.norm(cube_a_pos[:2] - target_pos[:2])
    target_height_error = abs(cube_a_pos[2] - (target_pos[2] + 0.045))
    return float(horizontal_distance), float(target_height_error)

@staticmethod
def _green_effective_shift(cube_b_pos, target_pos):
    shift = np.linalg.norm(cube_b_pos[:2] - target_pos[:2])
    return float(max(shift - 0.01, 0.0))

@staticmethod
def _green_grasp_event_reward(previous_grasped, grasped):
    return -0.50 if grasped and not previous_grasped else 0.0

@staticmethod
def _green_disturbance_reward(previous_shift, shift):
    increase = max(shift - previous_shift, 0.0)
    return -float(min(2.5 * increase, 0.25))
```

Generalize `_grasp_contacts()` to accept an optional object, while keeping the
existing no-argument cube-A behavior:

```python
def _grasp_contacts(self, obj=None):
    object_geoms = (self.cubeA if obj is None else obj).contact_geoms
    # Preserve the current per-gripper bilateral-contact selection.

def _cube_grasped(self, obj=None):
    left_contact, right_contact = self._grasp_contacts(obj)
    return left_contact and right_contact
```

- [ ] **Step 4: Run cube-B and existing multi-arm contact tests**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -k 'green or regress or reset_anchored or grasp_contacts or cube_grasped' -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit cube-B safeguards**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "test: specify Stack wrong-object safeguards"
```

### Task 3: Integrate augmented-state PBRS lifecycle

**Files:**
- Modify: `tests/test_environments/test_stack.py:1-175`
- Modify: `robosuite/environments/manipulation/stack.py:235-420`

- [ ] **Step 1: Replace the reward-only fixture with a snapshot-driven fixture**

Create `_reward_snapshot()` test data containing all `_stage_potential()` fields,
`green_effective_shift`, and physical success. Initialize `_reward_stage`,
`_prev_reward_potential`, `_prev_red_grasped`, `_prev_green_grasped`, and
`_prev_green_effective_shift` explicitly.

```python
def _reward_only_stack(*, snapshot=None, previous=0.0,
                       previous_stage=StackRewardStage.APPROACH,
                       reward_shaping=True, reward_scale=None,
                       previous_red_grasped=False,
                       previous_green_grasped=False,
                       previous_green_shift=0.0):
    env = Stack.__new__(Stack)
    env.reward_shaping = reward_shaping
    env.reward_scale = reward_scale
    env._reward_stage = previous_stage
    env._prev_reward_potential = previous
    env._prev_red_grasped = previous_red_grasped
    env._prev_green_grasped = previous_green_grasped
    env._prev_green_effective_shift = previous_green_shift
    current = _stage_kwargs(red_left_contact=False, red_right_contact=False,
                            red_grasped=False)
    current["green_effective_shift"] = 0.0
    if snapshot:
        current.update(snapshot)
    env._reward_snapshot = lambda: dict(current)
    env.timestep = 1
    env.horizon = 100
    env.ignore_done = False
    env.done = False
    return env
```

- [ ] **Step 2: Add failing lifecycle arithmetic tests**

Cover read-only idempotence, one-step stage commit, red event preservation,
green event/displacement addition, no-progress non-positive return, sparse
scaling, and successful-release exemption. Assert that `reward()` never mutates
the five caches while `_post_action()` updates all five once.

- [ ] **Step 3: Run lifecycle tests and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -k 'reward or post_action or progress or drop' -q
```

Expected: failures because `_compute_reward()` still uses `max(staged_rewards)`
and the old single grasp cache.

- [ ] **Step 4: Implement snapshot, potential, and commit-once reward flow**

`_reward_snapshot()` must use live cube A only for reach/lift, the reset target
for alignment/placement, and live cube B only for misuse detection, disturbance,
and physical success. `_compute_reward()` must:

```python
snapshot = self._reward_snapshot()
previous_stage = getattr(self, "_reward_stage", StackRewardStage.APPROACH)
stage = self._next_reward_stage(previous_stage, **snapshot)
potential = self._stage_potential(stage, **snapshot)
reward = STACK_SUCCESS_REWARD if snapshot["success"] else 0.0
if self.reward_shaping:
    if self._prev_reward_potential is not None:
        reward += REWARD_SHAPING_GAMMA * potential - self._prev_reward_potential
    if self._prev_red_grasped is not None:
        reward += self._grasp_event_reward(
            self._prev_red_grasped, snapshot["red_grasped"], snapshot["success"]
        )
    if self._prev_green_grasped is not None:
        reward += self._green_grasp_event_reward(
            self._prev_green_grasped, snapshot["green_grasped"]
        )
    if self._prev_green_effective_shift is not None:
        reward += self._green_disturbance_reward(
            self._prev_green_effective_shift,
            snapshot["green_effective_shift"],
        )
    if update_reward_state:
        self._reward_stage = stage
        self._prev_reward_potential = potential
        self._prev_red_grasped = snapshot["red_grasped"]
        self._prev_green_grasped = snapshot["green_grasped"]
        self._prev_green_effective_shift = snapshot["green_effective_shift"]
```

Keep final `reward_scale / 2.0` scaling unchanged. Make `_reward_potential()`
derive the candidate stage read-only from one snapshot. Keep `staged_rewards()`
as a diagnostic three-tuple `(approach_potential, stage_potential,
success_potential)` but do not take its maximum in the reward path.

- [ ] **Step 5: Run lifecycle and all pure reward tests**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -q
```

Expected: pure and fixture tests pass; reset assertions added in Task 4 may still
be absent.

- [ ] **Step 6: Commit augmented-state reward integration**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "feat: gate Stack PBRS by ordered task stage"
```

### Task 4: Seed and clear every reward cache on reset

**Files:**
- Modify: `tests/test_environments/test_stack.py:330-390`
- Modify: `robosuite/environments/manipulation/stack.py:520-545`

- [ ] **Step 1: Strengthen real-environment reset tests**

For shaping mode assert:

```python
assert env._reward_stage is StackRewardStage.APPROACH
assert env._stack_reward_target_pos == pytest.approx(
    env.sim.data.body_xpos[env.cubeB_body_id]
)
assert env._prev_reward_potential == pytest.approx(env._reward_potential())
assert env._prev_red_grasped == env._cube_grasped(env.cubeA)
assert env._prev_green_grasped == env._cube_grasped(env.cubeB)
assert env._prev_green_effective_shift == 0.0
```

For sparse mode assert the stage is `APPROACH`, the reset target is still
recorded for deterministic physical success diagnostics, and all four shaping
caches are `None`.

- [ ] **Step 2: Run reset tests and verify RED**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -k reset -q
```

Expected: failures for missing stage, target, and cube-B caches.

- [ ] **Step 3: Implement reset initialization after `sim.forward()`**

Always forward and copy cube B's propagated reset position, then initialize:

```python
self.sim.forward()
self._stack_reward_target_pos = np.array(
    self.sim.data.body_xpos[self.cubeB_body_id], copy=True
)
self._reward_stage = StackRewardStage.APPROACH
if self.reward_shaping:
    snapshot = self._reward_snapshot()
    self._prev_reward_potential = self._stage_potential(self._reward_stage, **snapshot)
    self._prev_red_grasped = snapshot["red_grasped"]
    self._prev_green_grasped = snapshot["green_grasped"]
    self._prev_green_effective_shift = snapshot["green_effective_shift"]
else:
    self._prev_reward_potential = None
    self._prev_red_grasped = None
    self._prev_green_grasped = None
    self._prev_green_effective_shift = None
```

- [ ] **Step 4: Run Stack tests and verify GREEN**

Run:

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit reset lifecycle**

```bash
git add robosuite/environments/manipulation/stack.py tests/test_environments/test_stack.py
git commit -m "fix: seed strict Stack reward state on reset"
```

### Task 5: Document migration and run adversarial verification

**Files:**
- Modify: `CHANGELOG.md`
- Test: `tests/test_environments/test_stack.py`
- Reference: `/data/ChenZihan/project/HIRL/robotenv/stack_auto_collect_wrapper.py`

- [ ] **Step 1: Update release notes**

Record both the defect and the replacement: the previous unordered transport
potential could rise from about `0.189` to `1.456` when only cube B moved under
a fixed lifted cube A, allowing green-cube grasping, pushing, and skipped stages
to corrupt late training. Then document the strict sequence, reset-anchored
cube-B target, wrong-object grasp penalty, displacement penalty, and requirement
to re-record or relabel dense demonstrations.

- [ ] **Step 2: Run formatting and focused regression checks**

```bash
git diff --check
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py tests/test_environments/test_all_environments.py -q
```

Expected: no whitespace errors and all selected tests pass.

- [ ] **Step 3: Run the broader environment regression suite**

```bash
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments -q
```

Expected: all tests pass, or any unrelated environment/dependency failure is
reported with its complete output and separated from Stack results.

- [ ] **Step 4: Run adversarial and successful trajectory probes**

Use the real headless Stack environment plus the existing HIRL Stack
auto-controller. Verify from logged stage/potential transitions that:

```text
APPROACH -> LIFT -> ALIGN -> PLACE
```

is the only forward order in successful attempts. Run a scripted cube-B grasp
or displacement probe and assert its cumulative shaping return is non-positive.

- [ ] **Step 5: Review the final diff and commit documentation**

```bash
git diff --check
git status --short
git diff --stat HEAD~4..HEAD
git add CHANGELOG.md
git commit -m "docs: explain strict-stage Stack shaping"
```

Expected: only Stack reward code, focused tests, release notes, this plan, and
the approved design document differ from the branch baseline.

### Task 6: Rewrite the feature branch as one replacement commit

**Files:**
- Preserve: every verified file in the final worktree
- Rewrite: Git history after the merge-base with `main`

- [ ] **Step 1: Record and inspect the exact branch base**

```bash
base=$(git merge-base main HEAD)
git log --oneline --decorate "$base"..HEAD
git diff --stat "$base"..HEAD
```

Expected: `$base` is the commit from which
`stack-pbrs-guidance-redesign` split, and the range contains only feature-branch
reward work.

- [ ] **Step 2: Re-run the final verification before rewriting history**

```bash
git diff --check "$base"..HEAD
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -q
```

Expected: no whitespace errors and all Stack tests pass.

- [ ] **Step 3: Squash every branch commit into one replacement commit**

```bash
git reset --soft "$base"
git commit -m "feat: add strict-stage Stack PBRS guidance"
```

This history rewrite is explicitly authorized by the user. Do not push it.

- [ ] **Step 4: Verify the rewritten branch shape and content**

```bash
test "$(git rev-list --count main..HEAD)" -eq 1
git log --oneline --decorate main..HEAD
git diff --check main..HEAD
git status --short
/data/ChenZihan/miniforge3/envs/hilserl/bin/python -m pytest \
  tests/test_environments/test_stack.py -q
```

Expected: exactly one feature commit, clean worktree, no whitespace errors, and
all Stack tests pass.
