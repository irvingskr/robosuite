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
        requested_grippers = list(gripper_types) if isinstance(gripper_types, (list, tuple)) else [gripper_types]
        assert requested_grippers in (["default"], ["ArxGripper"]), "PegInsertion requires ArxGripper"
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
