# Changelog

## [Unreleased] - 2026-06-28
### Features
- Added an ARX-only single-arm `PegInsertion` environment with a square peg and table-mounted square socket.
- Added fixed or seeded-random socket placement selected through a module-level code constant.
- Added staged insertion rewards and geometric object-state observations.
- Changed `PegInsertion` dense reward shaping to use progress deltas from a total staged potential.
- Penalized misaligned insertion depth in `PegInsertion` dense reward to discourage pushing into the socket before alignment.
- Changed `Stack` dense reward shaping to add discounted PBRS progress, `0.99 * Phi(s') - Phi(s)`, to the sparse task reward.
- Reworked Stack lifting, horizontal alignment, and downward placement into continuous geometric potential stages.
- Moved the default `Stack` object sampling region forward and enabled a narrower seeded-random `PegInsertion` hole region.

### Design Rationale
- Kept the environment independent from `NutAssembly` because nut selection and two-peg behavior do not apply.
- Initialized the peg as a free body between the ARX fingers so grasp stability comes from MuJoCo contact rather than a weld constraint.
- Preserved the standard seven-dimensional ARX action interface while forcing the gripper component closed for compatibility with existing scripts and datasets.
- Used potential deltas for dense insertion reward so static poses no longer keep producing shaping reward.
- Sharpened the depth reward with an alignment-squared gate and negative feedback for depth progress while poorly aligned.
- Used the zero-based maximum staged Stack score as the potential so no-progress random rollouts do not accumulate a constant positive return; stationary progressed states retain the expected `gamma < 1` discount cost.
- Scaled lift height over 15 cm, sharpened horizontal alignment distance, and added placement descent progress so successful demonstrations receive mostly positive dense rewards without adding a constant baseline.
- Cached the initial Stack potential after MuJoCo pose propagation so the first transition receives the correct PBRS term.

### Notes & Caveats
- The environment supports only `Arx5` with `ArxGripper`.
- The policy-provided gripper action is ignored.
- Free-body grasp stability depends on the ARX finger collision geometry and MuJoCo contact parameters.
- Stack PBRS uses a fixed discount of `0.99`; shaped transitions can exceed `reward_scale` because sparse and shaping rewards are additive.
