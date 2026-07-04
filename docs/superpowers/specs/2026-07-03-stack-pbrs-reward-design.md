# Stack PBRS Reward Design

## Goal

Replace Stack's repeated absolute dense reward with potential-based reward shaping while preserving the sparse task reward and providing continuous progress through lifting, alignment, and placement.

## Current Behavior and Review

`Stack.staged_rewards()` produces three phase scores for reaching / grasping, lifting / aligning, and successful stacking. `Stack.reward()` currently returns the maximum phase score on every step when `reward_shaping=True`.

The phase ordering is suitable as a coarse progress measure, but returning the absolute score repeatedly rewards remaining in any high-scoring state. The original lift score also jumped from zero to one, its alignment distance lacked a useful spatial scale, and lowering cube A onto cube B produced no progress. PBRS fixes repeated absolute rewards; a continuous geometric stage potential fixes the missing progress signal.

## Reward Definition

Define the state potential directly from the existing staged rewards:

```text
Phi(s) = max(r_reach(s), r_lift(s), r_stack(s))
```

The lift / align / place stage uses:

```text
lift     = clip((z_A - (z_table + 0.03)) / 0.15, 0, 1)
align    = 1 - tanh(10 * ||xy_A - xy_B||)
place    = 1 - clip(max(z_A - (z_B + 0.045), 0) / 0.10, 0, 1)
progress = max(lift, align)
r_lift   = 0.5 + 0.5 * progress + 0.4 * align + 0.5 * align^2 * place
```

Alignment and placement are active only after cube A is lifted. The stage is active only while cube A is grasped or above the task's four-centimeter lift threshold, preventing contact jitter from activating lift progress. It is bounded below the stack-success score of `2.0`.

With `gamma < 1`, a stationary progressed state receives the expected discount
cost `(gamma - 1) * Phi(s)`. Do not remove that cost by subtracting the success
score from every potential: that gauge adds a constant positive reward on every
nonterminal step and makes a 201-step no-progress HIRL rollout accumulate a
return near 2.0.

For a transition from `s` to `s'`, use discount `gamma = 0.99`:

```text
F(s, s') = 0.99 * Phi(s') - Phi(s)
```

The raw reward is:

```text
r_raw = r_sparse + F(s, s')    if reward_shaping is enabled
r_raw = r_sparse               otherwise
```

where `r_sparse` is `2.0` when stacking succeeds and `0.0` otherwise. Existing normalization remains unchanged:

```text
r = r_raw * reward_scale / 2.0
```

If `reward_scale` is `None`, return `r_raw` directly.

## State Lifecycle

- After reset has fully propagated object poses through MuJoCo, cache the initial `Phi(s0)`. This ensures the first environment transition receives the correct PBRS term.
- `reward()` computes a reward without mutating the cached potential so diagnostic calls are idempotent.
- `_post_action()` computes the transition reward once and then updates the cache to `Phi(s')`.
- Sparse-only behavior does not depend on or mutate potential state.

This follows PegInsertion's split between a read-only reward computation and the once-per-step state update, with the addition of the PBRS discount and correct initial-potential caching.

## Empirical Validation

Using the HIRL Stack environment with its 201-step limit, 10 uniformly random failed rollouts produced a mean undiscounted return near zero (`-0.00077`). Thirty successful auto-collected demonstrations produced `52.47%` positive steps, a mean dense-only return of `0.5385`, and a mean total return of `1.5385`. Individual stationary grasp / release steps may retain a negative discount cost, but successful trajectories have a positive median step reward and random episodes do not accumulate a positive baseline.

## Compatibility

- Preserve `_check_success()` and sparse success behavior while replacing the coarse lift / alignment stage with continuous geometry.
- Preserve the user's existing object-placement changes in `stack.py` and hole-placement changes in `peg_insertion.py`.
- Do not change PegInsertion in this task.
- Do not add a public constructor option for gamma; `0.99` is a task-level constant for this focused change.

## Tests

Add focused unit coverage for:

1. Shaped reward equals `r_sparse + 0.99 * Phi(s') - Phi(s)`.
2. Zero-progress states have zero potential and successful states have positive potential.
3. A stationary progressed state receives the expected discount cost.
4. A 201-step zero-progress rollout does not accumulate positive shaping return.
5. Sparse-only reward remains unchanged.
6. `reward_scale` preserves the existing `/ 2.0` normalization.
7. Repeated read-only `reward()` calls do not advance the cached potential.
8. `_post_action()` advances the cached potential exactly once.
9. Reset seeds the cache from the fully reset state, so the first transition is not forced to zero.
10. Lift, alignment, and downward placement form a strictly increasing potential sequence on successful geometry.
11. An untouched or sub-threshold jittered cube contributes no lift / align / place potential.
