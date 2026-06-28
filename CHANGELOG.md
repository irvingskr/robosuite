# Changelog

## [Unreleased] - 2026-06-28
### Features
- Added an ARX-only single-arm `PegInsertion` environment with a square peg and table-mounted square socket.
- Added fixed or seeded-random socket placement selected through a module-level code constant.
- Added staged insertion rewards and geometric object-state observations.

### Design Rationale
- Kept the environment independent from `NutAssembly` because nut selection and two-peg behavior do not apply.
- Initialized the peg as a free body between the ARX fingers so grasp stability comes from MuJoCo contact rather than a weld constraint.
- Preserved the standard seven-dimensional ARX action interface while forcing the gripper component closed for compatibility with existing scripts and datasets.

### Notes & Caveats
- The environment supports only `Arx5` with `ArxGripper`.
- The policy-provided gripper action is ignored.
- Free-body grasp stability depends on the ARX finger collision geometry and MuJoCo contact parameters.
