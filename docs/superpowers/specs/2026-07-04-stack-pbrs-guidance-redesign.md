# Stack PBRS Guidance Redesign

## Goal

Make Stack's dense reward useful for HIRL learning from reaching through
grasping, lifting, alignment, placement, and release, while retaining the
existing discounted PBRS core and never paying a positive reward merely for
remaining in a no-progress state.

## Diagnosis

The current reward is:

```text
r = r_sparse + 0.99 * Phi(s') - Phi(s)
Phi(s) = max(r_reach, r_lift, r_stack)
```

This arithmetic is correct, but the potential is too coarse around the skills
that HIRL must learn first. The existing potential has no partial-contact
state, gives only a single binary increase when a full grasp is detected, and
uses a reach signal whose median positive change is about `1e-4` in the
recorded controller trajectories.

The 30 successful dense demonstrations in
`stack_pri_dense_30_raw_successes_2026-07-04_19-00-55.pkl` contain 3,700
transitions. Of those transitions, 46.3% have negative rewards. During the
grasp phase, 95% are negative: acquiring the grasp produces one large positive
transition, followed by 19 stationary steps at `-0.0025` each. This is the
expected `-0.01 * Phi(s)` discount cost, but it leaves the critic with a weak
and discontinuous signal for reaching and contact acquisition.

The redesign therefore changes the state potential and adds paired grasp/drop
events. It does not add a constant reward offset or a repeated holding bonus.

## Approaches Considered

### Pure PBRS with a richer potential

This preserves policy invariance exactly and improves geometric guidance, but
full grasp acquisition remains only a potential jump and can still be
underrepresented in mixed replay.

### PBRS plus paired physical events (selected)

Keep the existing sparse-plus-PBRS formula, add partial finger contacts to the
state potential, add a one-time grasp-acquisition bonus, and apply a larger
penalty if that grasp is lost before success. The paired events make the skill
boundary visible while preventing repeated regrasping from becoming a reward
cycle.

These event terms mean the complete reward is not pure PBRS, but the PBRS core
and its discount remain unchanged. Every added term corresponds to an observed
physical transition.

### Copy Lift's action penalties

Lift penalizes action magnitude, action changes, and repeated closing without
a grasp. This is not selected because those penalties can exceed Stack's early
reach progress, and the open/close action sign is not consistent across the
supported grippers and the HIRL ARX wrapper. Stack should first regularize
physical outcomes rather than controller-specific commands.

## Reward Definition

The unscaled reward is:

```text
r_raw = r_sparse + 0.99 * Phi(s') - Phi(s) + r_event
```

As before, `r_sparse` is `2.0` on successful stacking and zero otherwise.
Existing scaling remains unchanged:

```text
r = r_raw * reward_scale / 2.0
```

When `reward_scale` is `None`, return `r_raw`.

### Reach and contact stage

Let `d_eef` be the minimum end-effector-to-cube-A-center distance. Let
`I_left` and `I_right` indicate whether any left or right fingerpad geom is in
contact with cube A.

```text
reach = 1 - tanh(5 * d_eef)
contact = 0.5 * (I_left + I_right)
r_reach_contact = 0.35 * reach + 0.25 * contact
```

Reducing the distance scale from 10 to 5 spreads a useful gradient over the
ARX arm's approach range. A single finger contact contributes `0.125`; a
bilateral contact contributes `0.25`. A stationary contact does not continue
paying a bonus because it is part of `Phi`, not an absolute per-step reward.

The bilateral contact definition remains identical to `_check_grasp()`.

### Lift, alignment, and placement stage

Define continuous progress values:

```text
lift = clip((z_A - (z_table + 0.03)) / 0.12, 0, 1)
align = 1 - tanh(10 * ||xy_A - xy_B||)    if z_A > z_table + 0.04 else 0
z_target = z_B + 0.045
place = 1 - clip(abs(z_A - z_target) / 0.10, 0, 1)
```

The placement term uses absolute height error. This fixes the current behavior
where every position below the target height receives full placement progress.

When cube A is grasped or above the four-centimeter lift threshold, define:

```text
r_transport = r_reach_contact
              + 0.55 * max(lift, align)
              + 0.40 * align
              + 0.25 * align^2 * place
```

Otherwise, `r_transport` is zero. Cap `r_transport` at `1.8`, below the `2.0`
success potential. Using `max(lift, align)` avoids erasing transport progress
when a correctly aligned cube descends toward cube B.

### Success and total potential

```text
r_stack = 2.0 if the existing strict stack-success condition holds, else 0
Phi(s) = max(r_reach_contact, r_transport, r_stack)
```

This retains the existing staged maximum and the exact
`0.99 * Phi(s') - Phi(s)` PBRS calculation.

### Event terms

Track whether cube A was grasped on the previous environment transition:

```text
+0.25  when bilateral grasp changes false -> true
-0.35  when bilateral grasp changes true -> false before task success
 0.00  otherwise
```

A release that completes the stack is exempt from the drop penalty. Since the
loss penalty is larger than the acquisition bonus, repeatedly grasping and
dropping cannot produce a positive event-reward cycle. Losing contact also
reduces the potential, so a bad drop receives both the PBRS loss of progress
and the explicit event penalty.

## No-Progress Behavior

There is no constant shift, living bonus, or repeated grasp-maintenance bonus.
For a stationary state:

```text
r_event = 0
F(s, s) = (0.99 - 1) * Phi(s) <= 0
```

An untouched state receives zero or a small negative discount cost, never a
positive reward. A stationary progressed state receives the existing PBRS
discount cost. A movement toward the cube, new finger contact, upward cube
motion, improved alignment, or improved target height is physical progress and
may receive a positive transition reward.

## State Lifecycle

- `_reward_potential()` remains a read-only state calculation.
- `reward()` computes the PBRS and event terms without mutating cached state,
  so repeated diagnostic calls are idempotent.
- `_post_action()` computes one transition reward and then updates both the
  previous potential and previous-grasp caches exactly once.
- `_reset_internal()` calls `sim.forward()`, seeds the initial potential from
  the fully propagated reset state, and seeds previous grasp state from the
  same state.
- Sparse-only mode neither depends on nor mutates shaping caches.

## Compatibility and Scope

- Preserve the existing sparse success condition, reward scaling, Stack object
  placement ranges, observations, and public constructor.
- Keep `REWARD_SHAPING_GAMMA = 0.99`, matching HIRL's model discount.
- Do not change PegInsertion or Lift.
- Do not add controller-specific action regularization in this change.
- Existing dense demonstrations must be re-recorded or replay-relabelled after
  this change; mixing old dense labels with the new online reward is invalid.
- HIRL training must set `env.sim.reward_shaping=true` and load the newly
  labelled dense demonstrations. The currently checked-out HIRL configuration
  is sparse and will not exercise this reward.

## Tests

Add focused coverage for:

1. Left and right finger contacts contribute independently, and bilateral
   contact matches the grasp state.
2. Far, near, one-finger contact, bilateral grasp, lift, alignment, placement,
   and stack form a strictly increasing representative potential sequence.
3. Placement progress decreases both above and below the target height.
4. A stationary untouched state receives a non-positive shaping reward.
5. A stationary progressed state receives a non-positive PBRS discount cost.
6. A grasp transition receives the one-time bonus; repeated read-only reward
   calls do not repeat or consume it.
7. A pre-success grasp loss receives the drop penalty, while a successful
   release does not.
8. Every complete grasp-then-drop event pair has negative net event reward.
9. `_post_action()` updates potential and grasp state once per transition.
10. Reset seeds both caches from the fully forwarded initial simulation state.
11. Sparse reward and `reward_scale / 2.0` behavior remain unchanged.

## Empirical Acceptance

After unit tests pass, collect successful HIRL auto-controller trajectories
with shaping enabled and compare them with the recorded baseline:

- every bilateral grasp acquisition transition is positive;
- every invalid grasp-loss transition is negative;
- median positive approach reward exceeds the previous approximately
  `0.000166` median;
- mean successful preterminal return exceeds the previous `0.4854` baseline;
- a stationary zero-action rollout has no positive rewards and a non-positive
  undiscounted shaping return;
- sparse success remains exactly `1.0` after the default reward scaling.
