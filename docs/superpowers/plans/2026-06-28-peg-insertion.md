# ARX Peg Insertion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ARX-only `PegInsertion` environment with a physically grasped square peg, table-mounted square socket, forced-closed gripper, staged reward, state observations, and deterministic fixed/random socket placement.

**Architecture:** Implement two reusable XML-backed object models and a standalone `ManipulationEnv` modeled after `NutAssembly`. Keep the standard ARX action shape while overriding the gripper component, and centralize geometric metrics so observations, reward, and success use identical semantics.

**Tech Stack:** Python 3.10, NumPy, MuJoCo through robosuite, MJCF/XML, pytest, Sphinx RST.

---

## File Map

- Create `robosuite/models/assets/objects/square-peg.xml`: free square peg collision and visual model.
- Create `robosuite/models/assets/objects/square-hole.xml`: fixed four-wall socket and base flange.
- Modify `robosuite/models/objects/xml_objects.py`: object wrappers and important sites.
- Modify `robosuite/models/objects/__init__.py`: public object exports.
- Create `robosuite/environments/manipulation/peg_insertion.py`: complete environment behavior.
- Modify `robosuite/__init__.py`: environment registration import.
- Create `tests/test_environments/test_peg_insertion.py`: focused regression and integration tests.
- Modify `docs/source/robosuite.environments.manipulation.rst`: API documentation entry.
- Create `CHANGELOG.md`: version-level feature record.

### Task 1: Square Peg and Socket Object Models

**Files:**
- Create: `robosuite/models/assets/objects/square-peg.xml`
- Create: `robosuite/models/assets/objects/square-hole.xml`
- Modify: `robosuite/models/objects/xml_objects.py:90-160`
- Modify: `robosuite/models/objects/__init__.py:4-22`
- Test: `tests/test_environments/test_peg_insertion.py`

- [ ] **Step 1: Write failing object contract tests**

Create `tests/test_environments/test_peg_insertion.py` with:

```python
import numpy as np

from robosuite.models.objects import SquareHoleObject, SquarePegObject


def _collision_geoms(obj):
    return [geom for geom in obj.get_obj().iter("geom") if geom.get("group") == "0"]


def test_square_peg_object_contract():
    peg = SquarePegObject(name="peg")

    assert len(peg.joints) == 1
    assert set(peg.important_sites) >= {"center", "top", "bottom"}
    geoms = _collision_geoms(peg)
    assert len(geoms) == 1
    assert geoms[0].get("type") == "box"
    assert np.allclose(np.fromstring(geoms[0].get("size"), sep=" "), [0.02, 0.02, 0.05])


def test_square_hole_object_contract():
    hole = SquareHoleObject(name="hole")

    assert hole.joints == []
    assert set(hole.important_sites) >= {"mouth", "bottom", "axis"}
    geoms = _collision_geoms(hole)
    assert len(geoms) == 5
    assert np.allclose(hole.bottom_offset, [0.0, 0.0, 0.0])
    assert np.allclose(hole.top_offset, [0.0, 0.0, 0.065])
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: collection fails because `SquareHoleObject` and `SquarePegObject` are not exported.

- [ ] **Step 3: Add the peg MJCF asset**

Create `robosuite/models/assets/objects/square-peg.xml`:

```xml
<mujoco model="square-peg">
  <asset>
    <material name="peg_green" rgba="0.12 0.55 0.20 1"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="peg" type="box" size="0.02 0.02 0.05" group="0"
              material="peg_green" density="250" friction="2.0 0.01 0.001"
              solimp="0.95 0.99 0.001" solref="0.01 1"/>
        <site name="center_site" pos="0 0 0" size="0.003" rgba="0 0 0 0"/>
      </body>
      <site name="bottom_site" pos="0 0 -0.05" size="0.003" rgba="0 0 0 0"/>
      <site name="top_site" pos="0 0 0.05" size="0.003" rgba="0 0 0 0"/>
      <site name="horizontal_radius_site" pos="0.02 0.02 0" size="0.003" rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
```

- [ ] **Step 4: Add the square socket MJCF asset**

Create `robosuite/models/assets/objects/square-hole.xml`:

```xml
<mujoco model="square-hole">
  <asset>
    <material name="hole_green" rgba="0.12 0.55 0.20 1"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="base" type="box" pos="0 0 0.0025" size="0.045 0.045 0.0025"
              group="0" material="hole_green" friction="1.0 0.01 0.001"/>
        <geom name="wall_pos_x" type="box" pos="0.028 0 0.035" size="0.005 0.033 0.03"
              group="0" material="hole_green" friction="1.0 0.01 0.001"/>
        <geom name="wall_neg_x" type="box" pos="-0.028 0 0.035" size="0.005 0.033 0.03"
              group="0" material="hole_green" friction="1.0 0.01 0.001"/>
        <geom name="wall_pos_y" type="box" pos="0 0.028 0.035" size="0.023 0.005 0.03"
              group="0" material="hole_green" friction="1.0 0.01 0.001"/>
        <geom name="wall_neg_y" type="box" pos="0 -0.028 0.035" size="0.023 0.005 0.03"
              group="0" material="hole_green" friction="1.0 0.01 0.001"/>
        <site name="mouth_site" pos="0 0 0.065" size="0.003" rgba="0 0 0 0"/>
        <site name="axis_site" pos="0 0 0.075" size="0.003" rgba="0 0 0 0"/>
      </body>
      <site name="bottom_site" pos="0 0 0" size="0.003" rgba="0 0 0 0"/>
      <site name="top_site" pos="0 0 0.065" size="0.003" rgba="0 0 0 0"/>
      <site name="horizontal_radius_site" pos="0.045 0.045 0" size="0.003" rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
```

- [ ] **Step 5: Add and export the XML object wrappers**

Add after `RoundNutObject` in `robosuite/models/objects/xml_objects.py`:

```python
class SquarePegObject(MujocoXMLObject):
    """Square peg used by the single-arm PegInsertion task."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-peg.xml"),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        sites = super().important_sites
        sites.update(
            {
                "center": self.naming_prefix + "center_site",
                "top": self.naming_prefix + "top_site",
                "bottom": self.naming_prefix + "bottom_site",
            }
        )
        return sites


class SquareHoleObject(MujocoXMLObject):
    """Fixed table-mounted square socket used by PegInsertion."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/square-hole.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        sites = super().important_sites
        sites.update(
            {
                "mouth": self.naming_prefix + "mouth_site",
                "bottom": self.naming_prefix + "bottom_site",
                "axis": self.naming_prefix + "axis_site",
            }
        )
        return sites
```

Add both names to the import tuple in `robosuite/models/objects/__init__.py`:

```python
    SquarePegObject,
    SquareHoleObject,
```

- [ ] **Step 6: Run the object tests**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit the object models**

```bash
git add robosuite/models/assets/objects/square-peg.xml robosuite/models/assets/objects/square-hole.xml robosuite/models/objects/xml_objects.py robosuite/models/objects/__init__.py tests/test_environments/test_peg_insertion.py
git commit -m "feat: add square peg insertion objects"
```

### Task 2: Environment Scene, Registration, and Configuration Guard

**Files:**
- Create: `robosuite/environments/manipulation/peg_insertion.py`
- Modify: `robosuite/__init__.py:4-14`
- Test: `tests/test_environments/test_peg_insertion.py`

- [ ] **Step 1: Add failing registration and configuration tests**

Append to `tests/test_environments/test_peg_insertion.py`:

```python
import pytest

import robosuite as suite


def _make_env(**kwargs):
    config = dict(
        env_name="PegInsertion",
        robots="Arx5",
        gripper_types="ArxGripper",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        hard_reset=False,
    )
    config.update(kwargs)
    return suite.make(**config)


def test_peg_insertion_is_registered_and_builds():
    assert "PegInsertion" in suite.ALL_ENVIRONMENTS
    env = _make_env()
    try:
        assert env.peg.root_body in env.sim.model.body_names
        assert env.hole.root_body in env.sim.model.body_names
    finally:
        env.close()


def test_peg_insertion_rejects_unsupported_robot():
    with pytest.raises(AssertionError, match="only supports Arx5"):
        _make_env(robots="Panda", gripper_types="ArxGripper")


def test_peg_insertion_rejects_unsupported_gripper():
    with pytest.raises(AssertionError, match="requires ArxGripper"):
        _make_env(gripper_types=None)
```

- [ ] **Step 2: Run the registration tests and verify failure**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: the object tests pass and registration tests fail because `PegInsertion` is absent.

- [ ] **Step 3: Create the initial environment implementation**

Create `robosuite/environments/manipulation/peg_insertion.py` with the constructor, scene, references, minimal reset, and validation below. Later tasks add control, metrics, reward, and observations to this class.

```python
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.grippers import ArxGripper
from robosuite.models.objects import SquareHoleObject, SquarePegObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor


RANDOMIZE_HOLE_POSITION = False
FIXED_HOLE_XY = np.array([0.10, 0.00])
HOLE_X_RANGE = (0.05, 0.15)
HOLE_Y_RANGE = (-0.10, 0.10)

PREGRASP_GRIPPER_QPOS = 0.0195
PEG_HALF_LENGTH = 0.05
PEG_GRASP_OVERLAP = 0.03
SUCCESS_DEPTH = 0.04
SUCCESS_XY_ERROR = 0.003
SUCCESS_ANGLE = np.deg2rad(5.0)
ALIGNMENT_ANGLE = np.deg2rad(10.0)


class PegInsertion(ManipulationEnv):
    """ARX-only square peg insertion task with a physical pre-grasp."""

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.82),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )
        assert isinstance(self.robots[0].gripper["right"], ArxGripper), "PegInsertion requires ArxGripper"

    def _check_robot_configuration(self, robots):
        names = [robots] if isinstance(robots, str) else list(robots)
        assert names == ["Arx5"], "PegInsertion only supports Arx5"
        super()._check_robot_configuration(robots)

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self.peg = SquarePegObject(name="peg")
        self.hole = SquareHoleObject(name="hole")
        self.hole.set_pos([FIXED_HOLE_XY[0], FIXED_HOLE_XY[1], self.table_offset[2]])
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.peg, self.hole],
        )

    def _setup_references(self):
        super()._setup_references()
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        self.peg_joint = self.peg.joints[0]
        self.peg_qvel_addr = self.sim.model.get_joint_qvel_addr(self.peg_joint)
        self.peg_center_site_id = self.sim.model.site_name2id(self.peg.important_sites["center"])
        self.peg_bottom_site_id = self.sim.model.site_name2id(self.peg.important_sites["bottom"])
        self.hole_mouth_site_id = self.sim.model.site_name2id(self.hole.important_sites["mouth"])
        self.hole_axis_site_id = self.sim.model.site_name2id(self.hole.important_sites["axis"])
        gripper = self.robots[0].gripper["right"]
        self.right_finger_geom_id = self.sim.model.geom_name2id(gripper.important_geoms["right_fingerpad"][0])
        self.left_finger_geom_id = self.sim.model.geom_name2id(gripper.important_geoms["left_fingerpad"][1])

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.model.body_pos[self.hole_body_id] = np.array(
            [FIXED_HOLE_XY[0], FIXED_HOLE_XY[1], self.table_offset[2]]
        )
        self.sim.data.set_joint_qpos(self.peg_joint, np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
        self.sim.forward()

    def reward(self, action=None):
        reward = float(self._check_success())
        return reward if self.reward_scale is None else reward * self.reward_scale

    def _check_success(self):
        return False
```

- [ ] **Step 4: Register the environment**

Add to `robosuite/__init__.py` beside the other manipulation imports:

```python
from robosuite.environments.manipulation.peg_insertion import PegInsertion
```

- [ ] **Step 5: Run registration and configuration tests**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: all five tests pass.

- [ ] **Step 6: Commit the scene and registration**

```bash
git add robosuite/environments/manipulation/peg_insertion.py robosuite/__init__.py tests/test_environments/test_peg_insertion.py
git commit -m "feat: register ARX peg insertion environment"
```

### Task 3: Physical Pre-Grasp and Forced-Closed Control

**Files:**
- Modify: `robosuite/environments/manipulation/peg_insertion.py`
- Test: `tests/test_environments/test_peg_insertion.py`

- [ ] **Step 1: Add failing pre-grasp and action tests**

Append:

```python
def _peg_contacts(env):
    peg_geoms = {env.sim.model.geom_name2id(name) for name in env.peg.contact_geoms}
    pairs = {
        frozenset((env.sim.data.contact[i].geom1, env.sim.data.contact[i].geom2))
        for i in range(env.sim.data.ncon)
    }
    return {
        "left": any(frozenset((env.left_finger_geom_id, peg_geom)) in pairs for peg_geom in peg_geoms),
        "right": any(frozenset((env.right_finger_geom_id, peg_geom)) in pairs for peg_geom in peg_geoms),
    }


def test_reset_places_peg_between_fingers_and_grasps_it():
    env = _make_env()
    try:
        env.reset()
        pad_midpoint = 0.5 * (
            env.sim.data.geom_xpos[env.left_finger_geom_id]
            + env.sim.data.geom_xpos[env.right_finger_geom_id]
        )
        peg_center = env.sim.data.site_xpos[env.peg_center_site_id]
        assert np.linalg.norm(peg_center[:2] - pad_midpoint[:2]) < 0.005
        assert peg_center[2] < pad_midpoint[2]

        action = np.zeros(env.action_dim)
        env.step(action)
        assert _peg_contacts(env) == {"left": True, "right": True}
    finally:
        env.close()


@pytest.mark.parametrize("requested_gripper", [-1.0, 0.0, 1.0])
def test_pre_action_forces_close_without_mutating_input(requested_gripper):
    env = _make_env()
    try:
        env.reset()
        action = np.zeros(env.action_dim)
        action[-1] = requested_gripper
        original = action.copy()
        env._pre_action(action, policy_step=True)
        assert np.array_equal(action, original)
        assert np.all(env.robots[0].gripper["right"].current_action < 0.0)
    finally:
        env.close()
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: pre-grasp/contact and forced-action assertions fail against the placeholder reset.

- [ ] **Step 3: Implement forced gripper control**

Add to `PegInsertion`:

```python
    def _pre_action(self, action, policy_step=False):
        forced_action = np.array(action, dtype=float, copy=True)
        forced_action[-1] = -1.0
        super()._pre_action(forced_action, policy_step=policy_step)
```

- [ ] **Step 4: Replace the placeholder reset with physical pre-grasp initialization**

Add this helper and replace `_reset_internal`:

```python
    def _set_pregrasp_pose(self):
        robot = self.robots[0]
        robot.set_gripper_joint_positions(
            np.full(2, PREGRASP_GRIPPER_QPOS, dtype=float),
            gripper_arm="right",
        )
        self.sim.forward()

        right_pos = np.array(self.sim.data.geom_xpos[self.right_finger_geom_id])
        left_pos = np.array(self.sim.data.geom_xpos[self.left_finger_geom_id])
        pad_midpoint = 0.5 * (right_pos + left_pos)
        pad_mat = np.array(self.sim.data.geom_xmat[self.right_finger_geom_id]).reshape(3, 3)
        pad_x_xy = pad_mat[:2, 0]
        yaw = np.arctan2(pad_x_xy[1], pad_x_xy[0])
        peg_quat_xyzw = T.euler2quat(np.array([0.0, 0.0, yaw]))
        peg_quat_wxyz = T.convert_quat(peg_quat_xyzw, to="wxyz")
        peg_center = pad_midpoint.copy()
        peg_center[2] -= PEG_HALF_LENGTH - PEG_GRASP_OVERLAP / 2.0

        self.sim.data.set_joint_qpos(
            self.peg_joint,
            np.concatenate([peg_center, peg_quat_wxyz]),
        )
        start, end = self.peg_qvel_addr
        self.sim.data.qvel[start:end] = 0.0
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.model.body_pos[self.hole_body_id] = np.array(
            [FIXED_HOLE_XY[0], FIXED_HOLE_XY[1], self.table_offset[2]]
        )
        self._set_pregrasp_pose()
```

- [ ] **Step 5: Run the pre-grasp and control tests**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: all tests pass. If MuJoCo contact settling exposes a calibration error, adjust only `PREGRASP_GRIPPER_QPOS` and `PEG_GRASP_OVERLAP`, then rerun until both finger contacts are present without initial penetration instability.

- [ ] **Step 6: Commit physical grasp control**

```bash
git add robosuite/environments/manipulation/peg_insertion.py tests/test_environments/test_peg_insertion.py
git commit -m "feat: initialize and hold peg in ARX gripper"
```

### Task 4: Fixed and Seeded-Random Socket Placement

**Files:**
- Modify: `robosuite/environments/manipulation/peg_insertion.py`
- Test: `tests/test_environments/test_peg_insertion.py`

- [ ] **Step 1: Add failing placement tests**

Add this module import and tests:

```python
import robosuite.environments.manipulation.peg_insertion as peg_module


def test_fixed_hole_position_is_restored(monkeypatch):
    monkeypatch.setattr(peg_module, "RANDOMIZE_HOLE_POSITION", False)
    env = _make_env()
    try:
        env.reset()
        first = env.sim.data.body_xpos[env.hole_body_id].copy()
        env.sim.model.body_pos[env.hole_body_id, :2] = [-0.2, 0.2]
        env.reset()
        second = env.sim.data.body_xpos[env.hole_body_id].copy()
        assert np.allclose(first[:2], peg_module.FIXED_HOLE_XY)
        assert np.allclose(second[:2], peg_module.FIXED_HOLE_XY)
    finally:
        env.close()


def test_random_hole_position_is_seeded_and_in_range(monkeypatch):
    monkeypatch.setattr(peg_module, "RANDOMIZE_HOLE_POSITION", True)
    env1 = _make_env(seed=7)
    env2 = _make_env(seed=7)
    try:
        sequence1 = []
        sequence2 = []
        for _ in range(3):
            env1.reset()
            env2.reset()
            sequence1.append(env1.sim.data.body_xpos[env1.hole_body_id, :2].copy())
            sequence2.append(env2.sim.data.body_xpos[env2.hole_body_id, :2].copy())
        assert np.allclose(sequence1, sequence2)
        assert all(peg_module.HOLE_X_RANGE[0] <= xy[0] <= peg_module.HOLE_X_RANGE[1] for xy in sequence1)
        assert all(peg_module.HOLE_Y_RANGE[0] <= xy[1] <= peg_module.HOLE_Y_RANGE[1] for xy in sequence1)
        assert not np.allclose(sequence1[0], sequence1[1])
    finally:
        env1.close()
        env2.close()
```

- [ ] **Step 2: Run placement tests and verify random-mode failure**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: fixed placement passes; random placement fails because reset still always uses the fixed coordinates.

- [ ] **Step 3: Implement placement selection**

Add and use this method in `_reset_internal` before `_set_pregrasp_pose()`:

```python
    def _reset_hole_position(self):
        if RANDOMIZE_HOLE_POSITION:
            xy = np.array(
                [
                    self.rng.uniform(*HOLE_X_RANGE),
                    self.rng.uniform(*HOLE_Y_RANGE),
                ]
            )
        else:
            xy = FIXED_HOLE_XY
        self.sim.model.body_pos[self.hole_body_id] = np.array([xy[0], xy[1], self.table_offset[2]])
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_hole_position()
        self._set_pregrasp_pose()
```

- [ ] **Step 4: Run placement and existing tests**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit placement modes**

```bash
git add robosuite/environments/manipulation/peg_insertion.py tests/test_environments/test_peg_insertion.py
git commit -m "feat: support fixed and random socket placement"
```

### Task 5: Metrics, Success, Staged Reward, and Object Observations

**Files:**
- Modify: `robosuite/environments/manipulation/peg_insertion.py`
- Test: `tests/test_environments/test_peg_insertion.py`

- [ ] **Step 1: Add a pose helper and failing metric/success tests**

Append:

```python
def _set_peg_pose(env, depth=0.04, xy_error=0.0, roll=0.0, yaw=0.0):
    quat_xyzw = T.euler2quat(np.array([roll, 0.0, yaw]))
    peg_axis = T.quat2mat(quat_xyzw) @ np.array([0.0, 0.0, 1.0])
    mouth = env.sim.data.site_xpos[env.hole_mouth_site_id].copy()
    bottom = mouth + np.array([xy_error, 0.0, -depth])
    center = bottom + PEG_HALF_LENGTH * peg_axis
    env.sim.data.set_joint_qpos(
        env.peg_joint,
        np.concatenate([center, T.convert_quat(quat_xyzw, to="wxyz")]),
    )
    env.sim.forward()


@pytest.mark.parametrize(
    "pose, expected",
    [
        ({}, True),
        ({"depth": 0.039}, False),
        ({"xy_error": 0.004}, False),
        ({"roll": np.deg2rad(6.0)}, False),
        ({"yaw": np.deg2rad(6.0)}, False),
        ({"yaw": np.deg2rad(90.0)}, True),
    ],
)
def test_success_boundaries(pose, expected):
    env = _make_env()
    try:
        env.reset()
        _set_peg_pose(env, **pose)
        assert env._check_success() is expected
    finally:
        env.close()
```

Add these imports at the top of the test file:

```python
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.peg_insertion import PEG_HALF_LENGTH
```

- [ ] **Step 2: Add failing reward and observation tests**

Append:

```python
def test_sparse_and_dense_rewards_are_scaled_and_bounded():
    sparse = _make_env(reward_shaping=False, reward_scale=2.0)
    dense = _make_env(reward_shaping=True, reward_scale=2.0)
    try:
        sparse.reset()
        dense.reset()
        _set_peg_pose(sparse, depth=0.04)
        _set_peg_pose(dense, depth=0.04)
        assert sparse.reward() == pytest.approx(2.0)
        assert dense.reward() == pytest.approx(2.0)

        _set_peg_pose(sparse, depth=0.02)
        _set_peg_pose(dense, depth=0.02)
        assert sparse.reward() == 0.0
        assert 0.0 < dense.reward() < 2.0
    finally:
        sparse.close()
        dense.close()


def test_object_observations_have_expected_names_and_shapes():
    env = _make_env()
    try:
        obs = env.reset()
        expected = {
            "peg_pos": (3,),
            "peg_quat": (4,),
            "hole_pos": (3,),
            "peg_to_hole_pos": (3,),
            "peg_bottom_to_hole_pos": (3,),
            "insertion_depth": (),
            "xy_error": (),
            "vertical_angle": (),
            "yaw_error": (),
        }
        for name, shape in expected.items():
            assert name in obs
            assert np.asarray(obs[name]).shape == shape
        assert "object-state" in obs
    finally:
        env.close()
```

- [ ] **Step 3: Run tests and verify metric/observation failures**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: success, dense reward, and object observation tests fail because the placeholder implementations remain.

- [ ] **Step 4: Implement shared insertion metrics and success**

Add:

```python
    @staticmethod
    def _square_yaw_error(peg_x, hole_x, hole_axis):
        peg_x = peg_x - np.dot(peg_x, hole_axis) * hole_axis
        hole_x = hole_x - np.dot(hole_x, hole_axis) * hole_axis
        peg_x /= np.linalg.norm(peg_x)
        hole_x /= np.linalg.norm(hole_x)
        signed = np.arctan2(np.dot(np.cross(hole_x, peg_x), hole_axis), np.dot(hole_x, peg_x))
        return abs((signed + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)

    def _compute_insertion_metrics(self):
        peg_bottom = np.array(self.sim.data.site_xpos[self.peg_bottom_site_id])
        hole_mouth = np.array(self.sim.data.site_xpos[self.hole_mouth_site_id])
        peg_mat = np.array(self.sim.data.body_xmat[self.peg_body_id]).reshape(3, 3)
        hole_mat = np.array(self.sim.data.body_xmat[self.hole_body_id]).reshape(3, 3)
        peg_axis = peg_mat[:, 2] / np.linalg.norm(peg_mat[:, 2])
        hole_axis = hole_mat[:, 2] / np.linalg.norm(hole_mat[:, 2])
        displacement = peg_bottom - hole_mouth
        insertion_depth = float(np.dot(hole_mouth - peg_bottom, hole_axis))
        planar = displacement - np.dot(displacement, hole_axis) * hole_axis
        xy_error = float(np.linalg.norm(planar))
        vertical_angle = float(np.arccos(np.clip(np.dot(peg_axis, hole_axis), -1.0, 1.0)))
        yaw_error = float(self._square_yaw_error(peg_mat[:, 0], hole_mat[:, 0], hole_axis))
        return {
            "peg_bottom": peg_bottom,
            "hole_mouth": hole_mouth,
            "insertion_depth": insertion_depth,
            "xy_error": xy_error,
            "vertical_angle": vertical_angle,
            "yaw_error": yaw_error,
        }

    def _check_success(self):
        metrics = self._compute_insertion_metrics()
        return bool(
            metrics["insertion_depth"] >= SUCCESS_DEPTH
            and metrics["xy_error"] <= SUCCESS_XY_ERROR
            and metrics["vertical_angle"] <= SUCCESS_ANGLE
            and metrics["yaw_error"] <= SUCCESS_ANGLE
        )
```

- [ ] **Step 5: Implement staged reward**

Replace `reward` and add `staged_rewards`:

```python
    def staged_rewards(self):
        metrics = self._compute_insertion_metrics()
        distance = np.linalg.norm(metrics["peg_bottom"] - metrics["hole_mouth"])
        approach = 0.25 * (1.0 - np.tanh(10.0 * distance))

        alignment = 0.0
        if distance <= 0.10:
            xy_score = 1.0 - np.tanh(50.0 * metrics["xy_error"])
            vertical_score = 1.0 - np.clip(metrics["vertical_angle"] / ALIGNMENT_ANGLE, 0.0, 1.0)
            yaw_score = 1.0 - np.clip(metrics["yaw_error"] / ALIGNMENT_ANGLE, 0.0, 1.0)
            alignment = 0.25 + 0.35 * np.mean([xy_score, vertical_score, yaw_score])

        insertion = 0.0
        if (
            metrics["xy_error"] <= 0.010
            and metrics["vertical_angle"] <= ALIGNMENT_ANGLE
            and metrics["yaw_error"] <= ALIGNMENT_ANGLE
        ):
            depth_progress = np.clip(metrics["insertion_depth"] / SUCCESS_DEPTH, 0.0, 1.0)
            insertion = 0.60 + 0.30 * depth_progress
        return float(approach), float(alignment), float(insertion)

    def reward(self, action=None):
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            reward = max(self.staged_rewards())
        else:
            reward = 0.0
        return reward if self.reward_scale is None else reward * self.reward_scale
```

- [ ] **Step 6: Implement object observations**

Add:

```python
    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        @sensor(modality=modality)
        def peg_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.peg_body_id])

        @sensor(modality=modality)
        def peg_quat(obs_cache):
            return T.convert_quat(np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw")

        @sensor(modality=modality)
        def hole_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.hole_mouth_site_id])

        @sensor(modality=modality)
        def peg_to_hole_pos(obs_cache):
            return hole_pos(obs_cache) - peg_pos(obs_cache)

        @sensor(modality=modality)
        def peg_bottom_to_hole_pos(obs_cache):
            metrics = self._compute_insertion_metrics()
            return metrics["hole_mouth"] - metrics["peg_bottom"]

        def metric_sensor(name):
            @sensor(modality=modality)
            def metric(obs_cache):
                return self._compute_insertion_metrics()[name]

            metric.__name__ = name
            return metric

        sensors = [
            peg_pos,
            peg_quat,
            hole_pos,
            peg_to_hole_pos,
            peg_bottom_to_hole_pos,
            metric_sensor("insertion_depth"),
            metric_sensor("xy_error"),
            metric_sensor("vertical_angle"),
            metric_sensor("yaw_error"),
        ]
        for observable_sensor in sensors:
            observables[observable_sensor.__name__] = Observable(
                name=observable_sensor.__name__,
                sensor=observable_sensor,
                sampling_rate=self.control_freq,
            )
        return observables
```

Remove the unused `OrderedDict` import after this implementation.

- [ ] **Step 7: Run focused tests**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: all focused tests pass.

- [ ] **Step 8: Commit task semantics**

```bash
git add robosuite/environments/manipulation/peg_insertion.py tests/test_environments/test_peg_insertion.py
git commit -m "feat: add peg insertion reward and observations"
```

### Task 6: Generic Environment Coverage, Smoke Coverage, Documentation, and Changelog

**Files:**
- Modify: `tests/test_environments/test_peg_insertion.py`
- Modify: `tests/test_environments/test_all_environments.py`
- Modify: `tests/test_environments/test_env_determinism.py`
- Modify: `docs/source/robosuite.environments.manipulation.rst`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Update generic environment tests for the ARX-only contract**

In both `tests/test_environments/test_all_environments.py` and `tests/test_environments/test_env_determinism.py`, replace:

```python
        for robot_name in ("Panda", "Sawyer", "Baxter", "GR1"):
```

with:

```python
        robot_names = ("Arx5",) if env_name == "PegInsertion" else ("Panda", "Sawyer", "Baxter", "GR1")
        for robot_name in robot_names:
```

This preserves existing coverage for every other environment and makes both generic suites construct `PegInsertion` only with its supported robot and default `ArxGripper`.

- [ ] **Step 2: Add a multi-step smoke regression test**

Append:

```python
def test_peg_insertion_zero_action_smoke():
    env = _make_env()
    try:
        obs = env.reset()
        for _ in range(10):
            obs, reward, done, info = env.step(np.zeros(env.action_dim))
            assert np.isfinite(reward)
            assert all(np.all(np.isfinite(value)) for value in obs.values())
            assert isinstance(done, (bool, np.bool_))
            assert isinstance(info, dict)
    finally:
        env.close()
```

- [ ] **Step 3: Run the smoke test**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py::test_peg_insertion_zero_action_smoke -q
```

Expected: pass only if the physical grasp remains numerically stable for ten control steps; otherwise fail with the first non-finite or simulation error.

- [ ] **Step 4: Add the Sphinx API entry**

Insert after the `nut_assembly` block in `docs/source/robosuite.environments.manipulation.rst`:

```rst
robosuite.environments.manipulation.peg\_insertion module
--------------------------------------------------------

.. automodule:: robosuite.environments.manipulation.peg_insertion
   :members:
   :undoc-members:
   :show-inheritance:
```

- [ ] **Step 5: Create the version changelog**

Create `CHANGELOG.md`:

```markdown
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
```

- [ ] **Step 6: Run focused tests and formatting checks**

Run:

```bash
pytest tests/test_environments/test_peg_insertion.py -q
python -m py_compile robosuite/environments/manipulation/peg_insertion.py robosuite/models/objects/xml_objects.py tests/test_environments/test_peg_insertion.py
git diff --check
```

Expected: all focused tests pass, Python compilation succeeds, and `git diff --check` prints no errors.

- [ ] **Step 7: Commit documentation and smoke coverage**

```bash
git add tests/test_environments/test_peg_insertion.py tests/test_environments/test_all_environments.py tests/test_environments/test_env_determinism.py docs/source/robosuite.environments.manipulation.rst CHANGELOG.md
git commit -m "docs: document ARX peg insertion environment"
```

### Task 7: Regression Verification

**Files:**
- Verify all files changed in Tasks 1-6.

- [ ] **Step 1: Run the focused environment suite**

```bash
pytest tests/test_environments/test_peg_insertion.py -q
```

Expected: all tests pass with no MuJoCo errors.

- [ ] **Step 2: Run existing environment coverage**

```bash
pytest tests/test_environments/test_all_environments.py -q
pytest tests/test_environments/test_env_determinism.py -q
```

Expected: both existing suites pass and construct `PegInsertion` with `Arx5`; all other environment/robot combinations remain unchanged.

- [ ] **Step 3: Inspect final state**

```bash
git status --short
git log --oneline -8
git diff HEAD~6 --stat
```

Expected: no uncommitted implementation changes; recent commits correspond to object models, environment registration, grasp control, placement, task semantics, and documentation.

- [ ] **Step 4: Perform one headless construction check**

```bash
python -c "import numpy as np; import robosuite as suite; env=suite.make('PegInsertion', robots='Arx5', gripper_types='ArxGripper', has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False); obs=env.reset(); print(env.action_dim, obs['peg_pos'], env._compute_insertion_metrics()); env.step(np.zeros(env.action_dim)); env.close()"
```

Expected: action dimension `7`, finite peg position and metrics, and a successful single step without exceptions.
