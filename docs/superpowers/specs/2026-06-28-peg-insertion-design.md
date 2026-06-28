# Single-Arm Peg Insertion Environment Design

## Summary

Add a new `PegInsertion` manipulation environment for the project's `Arx5` robot and `ArxGripper`. The task starts with a free-body square peg physically held between the gripper fingers. The policy aligns the peg with a fixed square socket on the table and inserts it while the environment forces the gripper to remain closed.

The environment follows the scene, observation, reward, and registration structure of `NutAssembly`, but it is an independent `ManipulationEnv` subclass. Inheriting from `NutAssembly` would retain irrelevant two-nut, two-peg, and single-object-mode behavior and would require overriding most of the parent implementation.

## Scope

The first version has these constraints:

- Robot support is limited to `Arx5` with `ArxGripper`.
- The peg is a real free-joint MuJoCo object. It is not welded or constrained to the end effector.
- The gripper remains closed for the whole episode. The external action space remains unchanged, but the environment ignores the gripper component supplied by the policy.
- The socket is fixed to the table. Its XY position can be fixed or randomized by editing a module-level constant; no constructor argument is added for this choice.
- The task succeeds while the gripper is still holding the peg. Release is not required.
- The square socket is assembled from primitive box collision geometries. No custom mesh is required.

## Files and Responsibilities

- `robosuite/environments/manipulation/peg_insertion.py`
  - Defines configuration constants and the `PegInsertion` environment.
  - Builds the scene, initializes the physical grasp, forces the gripper command closed, exposes object observations, computes insertion metrics, reward, and success.
- `robosuite/models/assets/objects/square-peg.xml`
  - Defines the free-body square peg geometry, material, and center/top/bottom sites.
- `robosuite/models/assets/objects/square-hole.xml`
  - Defines the fixed socket using four wall collision boxes and a base flange, plus mouth/bottom/axis sites.
- `robosuite/models/objects/xml_objects.py`
  - Adds `SquarePegObject` and `SquareHoleObject` wrappers and their important-site mappings.
- `robosuite/models/objects/__init__.py`
  - Exports the two object classes.
- `robosuite/__init__.py`
  - Imports `PegInsertion`, triggering metaclass registration and making `suite.make("PegInsertion", ...)` available.
- `tests/test_environments/test_peg_insertion.py`
  - Contains focused model, reset, control, randomization, metric, reward, success, and smoke tests.
- `docs/source/robosuite.environments.manipulation.rst`
  - Adds the new environment to generated manipulation-environment documentation.
- `CHANGELOG.md`
  - Records this version-level feature, its rationale, and its ARX/contact-stability limitations. Create the file if it does not exist.

## Object Geometry

### Peg

`SquarePegObject` uses a free joint and a single box collision geometry:

- Full size: `0.040 x 0.040 x 0.100 m`
- MuJoCo box half-size: `0.020 x 0.020 x 0.050 m`
- Local axis: positive local Z from bottom to top
- Material/color: opaque green, matching the supplied reference images
- Named sites:
  - `center_site` at `[0, 0, 0]`
  - `top_site` at `[0, 0, 0.050]`
  - `bottom_site` at `[0, 0, -0.050]`

The object wrapper exposes these sites as `center`, `top`, and `bottom` through `important_sites`.

### Square socket

`SquareHoleObject` has no joint and is fixed relative to the world. The socket consists of a thin base flange and four vertical wall boxes:

- Clear inner opening: `0.046 x 0.046 m`
- Inner half-width: `0.023 m`
- Wall thickness: `0.010 m`
- Usable depth from mouth to the top of the base: `0.060 m`
- Base flange full size: `0.090 x 0.090 x 0.005 m`
- Outer wall width: `0.066 m`
- Material/color: opaque green
- Named sites:
  - `mouth_site` at the center of the top opening
  - `bottom_site` at the center of the cavity floor
  - `axis_site` on the positive local Z axis above the mouth

The `0.046 m` opening provides `0.003 m` nominal clearance on each side of the `0.040 m` peg. The base flange closes the cavity bottom and visually matches the table-mounted reference object.

## Scene Construction and Placement

`PegInsertion` uses `TableArena` and the same table defaults as `NutAssembly`:

- `table_full_size=(0.8, 0.8, 0.05)`
- `table_friction=(1, 0.005, 0.0001)`
- `table_offset=(0, 0, 0.82)`

The environment adjusts the ARX base with the robot model's table offset, creates one `SquarePegObject` and one `SquareHoleObject`, and combines them with the arena and robot through `ManipulationTask`.

Socket placement is controlled by constants at the top of `peg_insertion.py`:

```python
RANDOMIZE_HOLE_POSITION = False
FIXED_HOLE_XY = np.array([0.10, 0.00])
HOLE_X_RANGE = (0.05, 0.15)
HOLE_Y_RANGE = (-0.10, 0.10)
```

In fixed mode, every reset restores `FIXED_HOLE_XY`. In random mode, every reset samples X and Y independently through `self.rng.uniform`, preserving seeded determinism. The socket orientation is never randomized. Since the socket has no joint, reset changes its root `model.body_pos` and calls `sim.forward()`.

## ARX-Only Validation

The environment accepts exactly one robot and rejects any model other than `Arx5`. It also verifies that the loaded right-arm gripper is an `ArxGripper`. Invalid configurations fail during environment construction with an error that states the supported robot and gripper instead of failing later during reset calibration.

The constructor retains the standard robosuite environment arguments. `gripper_types="default"` works because `Arx5.default_gripper` is `ArxGripper`; explicitly passing `gripper_types="ArxGripper"` also works.

## Physical Pre-Grasp Reset

The reset sequence runs after the normal robot reset:

1. Set both ARX finger slide joints to `PREGRASP_GRIPPER_QPOS = 0.0195 m`. This creates slight bilateral contact with the `0.040 m` peg.
2. Read the world positions and orientations of `right_finger_collision` and `left_finger_collision` after `sim.forward()`.
3. Use the midpoint of the two finger collision geometries as the lateral grasp center.
4. Keep the peg local Z axis aligned with world Z. Set its yaw from the gripper pad X direction projected into the world XY plane so the peg faces are parallel to the finger pads.
5. Position the peg so its upper `0.030 m` overlaps the finger-pad vertical span. With a `0.100 m` peg, this places the peg center approximately `0.035 m` below the finger-pad midpoint; the implementation derives the final pose from the live pad transforms rather than hard-coding a world pose.
6. Write the peg free-joint pose, zero its six joint velocities, and call `sim.forward()`.

No simulation time is advanced inside reset. The first external step applies the forced-close command before integration. The initial robot pose leaves the peg suspended above the table and spatially separated from the socket.

## Forced Gripper Control

The environment keeps the standard seven-dimensional ARX action interface for compatibility with existing controllers, collection scripts, and datasets. `PegInsertion._pre_action()` copies the caller-provided array, sets its final element to `-1.0`, and delegates to `RobotEnv._pre_action()`.

Copying avoids mutating a policy-owned action buffer. The arm's six action components pass through unchanged. The gripper component is intentionally ignored and always commands closing.

## Insertion Metrics and Success

The environment computes all task metrics from MuJoCo sites and body orientations:

- `peg_bottom`: world position of the peg's bottom site
- `hole_mouth`: world position of the socket's mouth site
- `hole_axis`: normalized socket local Z axis
- `peg_axis`: normalized peg local Z axis
- `insertion_depth = dot(hole_mouth - peg_bottom, hole_axis)`, clipped only when exposed as a reward feature
- `xy_error`: norm of the peg-bottom-to-mouth displacement projected onto the plane perpendicular to `hole_axis`
- `vertical_angle`: `arccos(clip(dot(peg_axis, hole_axis), -1, 1))`
- `yaw_error`: the smallest projected peg-to-socket yaw difference modulo `pi / 2`, because a square is invariant under 90-degree rotations

Success requires all of the following at the same simulation state:

- `insertion_depth >= 0.040 m`
- `xy_error <= 0.003 m`
- `vertical_angle <= 5 degrees`
- `yaw_error <= 5 degrees`

Gripper release and an explicit grasp-contact check are not part of success. The environment enforces closing throughout the episode, while success remains a geometric task condition.

## Reward

Sparse reward is `1.0` on success and `0.0` otherwise. If `reward_scale` is not `None`, the result is multiplied by `reward_scale`.

Dense reward is staged and non-negative. It remains below the terminal reward until the strict success condition is met:

1. Approach stage, maximum `0.25`:
   - `0.25 * (1 - tanh(10 * ||peg_bottom - hole_mouth||))`
2. Alignment stage, range `[0.25, 0.60]`, active only when the peg bottom is within `0.10 m` of the mouth:
   - XY score: `1 - tanh(50 * xy_error)`
   - Vertical score: `1 - clip(vertical_angle / 10 degrees, 0, 1)`
   - Yaw score: `1 - clip(yaw_error / 10 degrees, 0, 1)`
   - Alignment reward: `0.25 + 0.35 * mean(xy_score, vertical_score, yaw_score)`
3. Insertion stage, range `[0.60, 0.90]`, active only when `xy_error <= 0.010 m`, `vertical_angle <= 10 degrees`, and `yaw_error <= 10 degrees`:
   - `0.60 + 0.30 * clip(insertion_depth / 0.040, 0, 1)`
4. Terminal stage:
   - strict success overrides the staged value with `1.0`

The unscaled dense reward is the maximum active stage, following the staged-reward structure used by `NutAssembly`. The final value is multiplied by `reward_scale` when configured.

## Observations

Standard robot and camera observations are inherited. When `use_object_obs=True`, the environment adds sensors in the `object` modality:

- `peg_pos`: peg root position, shape `(3,)`
- `peg_quat`: peg quaternion in robosuite XYZW convention, shape `(4,)`
- `hole_pos`: hole mouth position, shape `(3,)`
- `peg_to_hole_pos`: `hole_mouth - peg root position`, shape `(3,)`
- `peg_bottom_to_hole_pos`: `hole_mouth - peg_bottom`, shape `(3,)`
- `insertion_depth`: scalar
- `xy_error`: scalar
- `vertical_angle`: scalar radians
- `yaw_error`: scalar radians

These sensors are included in the automatically composed `object-state` observation. Metric sensors call one shared metric helper so reward, success, and observation semantics cannot drift.

## Reset and Step Data Flow

```text
reset
  -> normal robot/table reset
  -> fixed or seeded-random socket placement
  -> finger qpos set to pre-grasp width
  -> peg pose derived from live finger collision transforms
  -> peg velocity cleared
  -> sim.forward()
  -> observations

step(policy_action)
  -> copy action
  -> force gripper element to -1
  -> normal ARX controller and MuJoCo integration
  -> shared insertion metrics
  -> staged or sparse reward
  -> observations and standard horizon termination
```

Dropping the peg does not trigger a special terminal state. It simply makes insertion success unreachable for that episode. Standard robosuite horizon and `ignore_done` behavior remain unchanged.

## Testing Strategy

Focused tests use `Arx5`, disable on-screen and off-screen renderers unless camera behavior is specifically needed, and close every environment instance.

1. Registration and validation
   - `PegInsertion` appears in `suite.ALL_ENVIRONMENTS` and constructs through `suite.make`.
   - Unsupported robots and grippers raise the intended configuration error.
2. Object model contract
   - Peg and socket XMLs compile.
   - Full dimensions and important-site names match this specification.
3. Pre-grasp reset
   - Peg is between the two finger collision centers.
   - Both finger collision geoms contact the peg after one zero-arm action step, during which closing is forced.
   - Peg bottom starts above the table and outside the socket.
4. Forced gripper action
   - Inputs with open, neutral, and close values all reach the robot control path with the final component equal to `-1.0`.
   - The input array remains unchanged.
5. Socket placement
   - Fixed mode restores the exact fixed XY position across resets.
   - Random mode stays within configured ranges.
   - Two environments with the same seed produce the same random socket sequence.
6. Metrics and success boundaries
   - A centered, vertical, yaw-aligned pose at `0.040 m` depth succeeds.
   - Depth below `0.040 m`, XY error above `0.003 m`, vertical angle above 5 degrees, and yaw error above 5 degrees each fail independently.
   - Yaw rotations of 90 degrees are treated as equivalent.
7. Reward
   - Sparse reward is binary and scales correctly.
   - Dense stage values are ordered, bounded by `1.0`, and terminal success returns exactly the scaled terminal value.
8. Smoke behavior
   - Reset and multiple zero-arm steps run without simulation errors or NaNs.
   - Required object observations have stable names and shapes.

Tests that modify the module-level randomization constant restore its original value in `finally` or through pytest monkeypatch isolation.

## Documentation and Changelog

The manipulation-environment API documentation will include `PegInsertion`. The root changelog entry will use the repository-required version-level format and state:

- Feature: ARX-only single-arm square peg insertion environment
- Rationale: independent environment modeled after `NutAssembly`, with a physical pre-grasp rather than a welded peg
- Caveats: gripper input is ignored, the environment supports only `Arx5`/`ArxGripper`, and free-body grasp stability depends on MuJoCo contact parameters

## Non-Goals

- Supporting Panda, Sawyer, or other gripper geometries
- Releasing or regrasping the peg
- Randomizing socket orientation
- Adding mesh-based or photorealistic assets
- Adding a constructor parameter for fixed versus random socket placement
- Early episode termination when the peg is dropped
- Reusing or changing `TwoArmPegInHole`
