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

    def _pre_action(self, action, policy_step=False):
        forced_action = np.array(action, dtype=float, copy=True)
        forced_action[-1] = -1.0
        super()._pre_action(forced_action, policy_step=policy_step)

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
        peg_quat_xyzw = T.mat2quat(T.euler2mat(np.array([0.0, 0.0, yaw])))
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

    def _check_success(self):
        metrics = self._compute_insertion_metrics()
        return bool(
            metrics["insertion_depth"] >= SUCCESS_DEPTH
            and metrics["xy_error"] <= SUCCESS_XY_ERROR
            and metrics["vertical_angle"] <= SUCCESS_ANGLE
            and metrics["yaw_error"] <= SUCCESS_ANGLE
        )

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
