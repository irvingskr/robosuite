# Stack PBRS Reward Design

## Goal

Replace Stack's repeated absolute dense reward with potential-based reward shaping while preserving the existing sparse task reward and staged reward semantics.

## Current Behavior and Review

`Stack.staged_rewards()` produces three phase scores for reaching / grasping, lifting / aligning, and successful stacking. `Stack.reward()` currently returns the maximum phase score on every step when `reward_shaping=True`.

The phase ordering is suitable as a coarse progress measure, but returning the absolute score repeatedly rewards remaining in any high-scoring state. The lift and grasp milestones are discontinuous and the alignment term has a weak distance gradient, but those characteristics do not invalidate the score as a bounded potential. This change intentionally preserves them to avoid expanding the scope into reward retuning.

## Reward Definition

Define the state potential from the existing staged rewards:

```text
Phi(s) = max(r_reach(s), r_lift(s), r_stack(s))
```

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

## Compatibility

- Keep `staged_rewards()` and `_check_success()` behavior unchanged.
- Preserve the user's existing object-placement changes in `stack.py` and hole-placement changes in `peg_insertion.py`.
- Do not change PegInsertion in this task.
- Do not add a public constructor option for gamma; `0.99` is a task-level constant for this focused change.

## Tests

Add focused unit coverage for:

1. Shaped reward equals `r_sparse + 0.99 * Phi(s') - Phi(s)`.
2. Sparse-only reward remains unchanged.
3. `reward_scale` preserves the existing `/ 2.0` normalization.
4. Repeated read-only `reward()` calls do not advance the cached potential.
5. `_post_action()` advances the cached potential exactly once.
6. Reset seeds the cache from the fully reset state, so the first transition is not forced to zero.

