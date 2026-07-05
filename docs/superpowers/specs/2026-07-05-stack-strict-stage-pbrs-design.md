# Stack Strict-Stage PBRS Design

## Goal

Make Stack's dense reward teach one valid sequence: approach and grasp cube A
(red), lift it to clearance, align it over cube B (green), lower it, and release
on a valid stack. A policy must not obtain later-stage reward by grasping or
moving cube B, dragging cube A across the table, or otherwise changing relative
geometry without completing the earlier physical stages.

The sparse Stack success condition, reward scaling, and PBRS discount remain
unchanged.

## Root Cause

The current transport potential opens when cube A is either grasped **or merely
above a four-centimeter height threshold**. Alignment and placement then depend
on cube A relative to cube B. Consequently, after cube A has been lifted or
slightly displaced, moving cube B underneath cube A improves the potential even
when the gripper is holding cube B instead of cube A. A representative pure
geometry check increases the transport potential from about `0.189` to `1.456`
solely by moving cube B under a fixed cube A.

The current `max(reach, transport, success)` structure also has no memory that
lift was completed before descent. Instantaneous geometry alone cannot
distinguish a valid post-lift descent from a cube that was dragged at table
height.

## Considered Approaches

### Require only a current cube-A grasp

Mask transport reward unless cube A is bilaterally grasped. This closes the
direct cube-B exploit but still allows a grasped cube A to collect alignment or
placement progress without first reaching a safe lift height. The stage
boundaries remain weak, so this is insufficient.

### Stateless geometric gates

Require lift height for alignment and alignment for placement using only the
current simulator state. This cannot represent the full task: after a valid
lift, cube A must descend below the lift threshold to be placed. Either descent
incorrectly closes the later stage, or a low aligned pose is allowed to skip
the lift stage.

### Regressible stage state plus PBRS (selected)

Maintain a small reward-stage state and compute the potential over the augmented
state `(simulator_state, reward_stage)`. Every forward transition requires a
specific physical predicate. Invalid grasp loss and alignment drift regress the
stage. The reward still uses the original PBRS difference, but the potential
now encodes the ordered task rather than an unordered maximum of geometry
scores.

## Ordered Reward Stages

Use four nonterminal stages:

```text
APPROACH -> LIFT -> ALIGN -> PLACE -> physical Stack success
```

Only one forward stage transition may occur per environment transition. This
prevents a single coincidental pose from skipping multiple prerequisites.

### APPROACH

This is the reset stage. Let `d_red` be the minimum end-effector-to-cube-A
distance, and let `I_left_red` and `I_right_red` represent fingerpad contacts
with cube A:

```text
reach_red = 1 - tanh(5 * d_red)
Phi_approach = 0.30 * reach_red
               + 0.10 * I_left_red
               + 0.10 * I_right_red
```

The range is `[0, 0.50]`. Bilateral cube-A contact advances the stage to
`LIFT`. Cube-B proximity or contact never contributes to this term.

### LIFT

This stage is valid only while cube A remains bilaterally grasped:

```text
lift = clip((z_A - (z_table + 0.02)) / 0.10, 0, 1)
Phi_lift = 0.50 + 0.45 * lift
```

The range is `[0.50, 0.95]`. Reaching `z_table + 0.12` advances to `ALIGN`.
Losing the cube-A grasp before success immediately regresses to `APPROACH`.

### ALIGN

The stage records that the clearance lift was completed. Record cube B's reset
position as the fixed dense-reward target `p_B_reset`; alignment progress is
active only while cube A remains high enough for safe transport:

```text
align = 1 - tanh(10 * ||xy_A - xy_B_reset||)
Phi_align = 0.95 + 0.40 * align      if z_A >= z_table + 0.10
            0.95                     otherwise
```

The range is `[0.95, 1.35]`. Horizontal error at or below `0.035 m`, while the
height guard holds, advances to `PLACE`. Lowering cube A before alignment
therefore provides no alignment progress.

### PLACE

Let the desired cube-A center height be `z_B_reset + 0.045`. While reset-target
horizontal error remains within `0.055 m`:

```text
place = 1 - clip(abs(z_A - (z_B_reset + 0.045)) / 0.10, 0, 1)
Phi_place = 1.35 + 0.35 * place
```

The range is `[1.35, 1.70]`. The absolute height error prevents overshooting
below the target from receiving full reward. If horizontal error exceeds
`0.055 m`, the stage regresses to `ALIGN`; the wider exit threshold provides
hysteresis and prevents threshold chatter. Losing the cube-A grasp regresses to
`APPROACH`, except when the same transition satisfies the existing physical
Stack success condition.

### SUCCESS

The existing strict success predicate is unchanged: cube A is released, remains
lifted, and contacts cube B. Its potential is `2.0`, and the sparse unscaled
success reward remains `2.0`.

## Cube-B Misuse

Finger contacts are computed independently for both cubes. Bilaterally grasping
cube B has three effects:

1. The valid task stage regresses to `APPROACH`.
2. The task-progress component of the current potential is forced to zero.
3. A one-time `-0.50` event penalty is applied on the false-to-true cube-B grasp
   transition.

Holding cube B does not repeatedly add a penalty, but it cannot generate task
progress. Releasing cube B adds no reward.

Use the same reset pose as the fixed dense-reward target and cache cube B's
previous effective displacement:

```text
green_shift = ||xy_B - xy_B_reset||
effective_shift = max(green_shift - 0.01, 0)
r_green_move = -min(2.5 * max(effective_shift - previous_effective_shift, 0), 0.25)
```

The one-centimeter dead band tolerates contact noise. Only a transition that
moves cube B farther from its reset pose is penalized. Holding it still produces
no repeated penalty, and moving it back does not refund reward. This event term
is deliberately not represented as a negative potential: a stationary negative
potential would create a small positive PBRS reward when `gamma < 1`. Valid
contact during final placement can cause small motion inside the dead band
without penalty.

Alignment and placement always use the reset target, not cube B's live pose.
Therefore moving cube B cannot improve the task potential in any stage. The
existing sparse success predicate still uses physical cube-to-cube contact, so
a slightly displaced but genuinely completed stack remains a success.

## Complete Reward

For shaping mode, derive `next_stage` from the previous stage and the current
physical state, calculate the augmented-state potential, and apply:

```text
r_raw = r_sparse
        + 0.99 * Phi(current_sim_state, next_stage)
        - Phi(previous_sim_state, previous_stage)
        + r_red_grasp_event
        + r_green_grasp_event
        + r_green_move
```

Preserve the existing red grasp events:

```text
+0.35  cube-A bilateral grasp false -> true
-0.45  cube-A bilateral grasp true -> false before success
```

Add the cube-B event:

```text
-0.50  cube-B bilateral grasp false -> true
```

Existing scaling remains:

```text
r = r_raw * reward_scale / 2.0
```

When `reward_scale` is `None`, return `r_raw`. Sparse-only mode remains
stateless and unchanged.

## State Lifecycle

- `reward()` derives the candidate next stage and reward without mutating any
  cache, so repeated diagnostic calls are idempotent.
- `_reward_potential()` reports the potential of the currently committed stage
  and never previews or advances another milestone.
- `_post_action()` commits the derived stage, potential, red-grasp state,
  green-grasp state, and effective cube-B displacement exactly once per
  simulator transition.
- `_reset_internal()` calls `sim.forward()`, records cube B's reset position,
  initializes the stage to `APPROACH`, and seeds all reward caches, including
  effective cube-B displacement, from the propagated reset state.
- The stage is reward-internal and does not change Stack observations or HIRL's
  policy input shape. It is part of the augmented environment state used by the
  shaping calculation.

## No-Progress and Anti-Cycling Properties

- There is no living, holding, or constant stage bonus added per step.
- At a stationary state, PBRS contributes `(0.99 - 1) * Phi`, which is zero or
  negative for ordinary nonnegative progress states.
- A red grasp followed by an invalid drop has negative net event reward
  (`0.35 - 0.45 = -0.10`) in addition to losing potential.
- A cube-B grasp has a direct negative event and cannot unlock any forward
  stage.
- Moving cube B cannot improve red reach, lift, alignment eligibility, or
  placement eligibility before the required red-cube stages have completed;
  increasing its reset-relative displacement adds a grounded negative event.

## Compatibility and Scope

- Preserve `REWARD_SHAPING_GAMMA = 0.99`, sparse success, `reward_scale`, object
  sampling, observations, and the public constructor.
- Change only Stack reward logic, its focused tests, and release notes.
- Do not add controller-action penalties; gripper action conventions vary and
  the physical-state penalties already target the observed failure mode.
- Dense demonstrations collected under the previous reward must be re-recorded
  or reward-relabelled. Mixing old dense labels with the new online reward is
  invalid.

## Tests

Focused tests must prove:

1. Stage transitions advance in order and at most once per environment step.
2. Alignment is unavailable before lift completion, including while cube A is
   aligned at table height.
3. Placement is unavailable before alignment completion.
4. A completed lift remains recorded during valid descent, while low transport
   before alignment earns no alignment progress.
5. Red grasp loss regresses to `APPROACH` and receives the existing penalty;
   successful release is exempt.
6. Alignment drift from `PLACE` regresses to `ALIGN` using hysteresis.
7. Grasping cube B resets progress, yields a one-time negative event, and never
   yields repeated reward while held.
8. Moving cube B underneath a fixed cube A cannot increase potential in any
   stage because the dense target is reset-anchored, and movement incurs a
   disturbance cost outside the dead band.
9. Stationary and no-progress rollouts do not accumulate positive reward.
10. Read-only reward calls remain idempotent and `_post_action()` commits state
    once.
11. Reset initializes stage, potential, both grasp histories, and cube-B reset
    position after simulation propagation.
12. Sparse reward and existing scaling remain unchanged.

## Empirical Acceptance

After unit and environment tests pass, replay a successful Stack controller
trajectory and inspect its transition rewards. The first positive transitions
for each later range must occur in the order `LIFT`, `ALIGN`, then `PLACE`.
Also run adversarial scripted traces that grasp or move cube B; none may produce
a positive cumulative shaping return. New HIRL training should use only dense
demonstrations labelled with this reward version.
