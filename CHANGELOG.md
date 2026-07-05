# Changelog

## [Unreleased] - 2026-06-28
### Features
- Added an ARX-only single-arm `PegInsertion` environment with a square peg and table-mounted square socket.
- Added fixed or seeded-random socket placement selected through a module-level code constant.
- Added staged insertion rewards and geometric object-state observations.
- Changed `PegInsertion` dense reward shaping to use progress deltas from a total staged potential.
- Penalized misaligned insertion depth in `PegInsertion` dense reward to discourage pushing into the socket before alignment.
- Changed `Stack` dense reward shaping to add discounted PBRS progress, `0.99 * Phi(s') - Phi(s)`, to the sparse task reward.
- Replaced Stack's unordered transport potential with a regressible sequence: approach red, grasp red, lift to clearance, align to the reset-anchored green target, place, and physically release onto green.
- Added a one-time wrong-object grasp penalty and a bounded green-cube displacement penalty; moving or holding the green cube cannot unlock red-cube task progress.
- Moved the default `Stack` object sampling region forward and enabled a narrower seeded-random `PegInsertion` hole region.

### Design Rationale
- Kept the environment independent from `NutAssembly` because nut selection and two-peg behavior do not apply.
- Initialized the peg as a free body between the ARX fingers so grasp stability comes from MuJoCo contact rather than a weld constraint.
- Preserved the standard seven-dimensional ARX action interface while forcing the gripper component closed for compatibility with existing scripts and datasets.
- Used potential deltas for dense insertion reward so static poses no longer keep producing shaping reward.
- Sharpened the depth reward with an alignment-squared gate and negative feedback for depth progress while poorly aligned.
- Used a zero-based ordered Stack stage as the potential so later geometry is unavailable until earlier physical milestones complete; stationary progressed states retain the expected `gamma < 1` discount cost.
- The superseded cumulative transport formula could increase from about `0.189` to `1.456` when only the green cube moved under a fixed lifted red cube. HIRL could therefore regress from lifting red to grasping or pushing green and skipping intended task stages.
- Added explicit stage memory because instantaneous geometry cannot distinguish valid descent after a completed lift from dragging a never-lifted cube across the table; grasp loss and placement drift regress that stage.
- Anchored dense alignment and placement to the green cube's reset pose so moving green cannot improve the potential, while keeping live cube-to-cube contact as the sparse success condition.
- Cached the initial Stack potential after MuJoCo pose propagation so the first transition receives the correct PBRS term.
- Kept Stack's no-progress PBRS reward non-positive while pairing grasp acquisition with a larger pre-success grasp-loss penalty to prevent reward cycling.

### Notes & Caveats
- The environment supports only `Arx5` with `ArxGripper`.
- The policy-provided gripper action is ignored.
- Free-body grasp stability depends on the ARX finger collision geometry and MuJoCo contact parameters.
- Stack PBRS uses a fixed discount of `0.99`; shaped transitions can exceed `reward_scale` because sparse and shaping rewards are additive.
- Stack dense demonstrations labelled with the superseded unordered reward must be re-recorded or relabelled before training with the strict-stage reward.
