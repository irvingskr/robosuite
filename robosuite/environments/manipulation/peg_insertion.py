import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.grippers import ArxGripper
from robosuite.models.objects import SquareHoleObject, SquarePegObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor


RANDOMIZE_HOLE_POSITION = True
FIXED_HOLE_XY = np.array([0.27, 0.00])
HOLE_POSITION_RADIUS = 0.05
HOLE_YAW_RANGE = np.pi / 4.0
AGENTVIEW_POSITION = np.array([0.6, 0.0, 1.35])
AGENTVIEW_QUATERNION = np.array([0.653, 0.271, 0.271, 0.653])

PREGRASP_GRIPPER_QPOS = 0.0195
INITIAL_CLEARANCE_ABOVE_HOLE = 0.01
PEG_HALF_LENGTH = 0.05
PEG_GRASP_OVERLAP = 0.03
SUCCESS_DEPTH = 0.04
SUCCESS_XY_ERROR = 0.003
ALIGNMENT_ANGLE = np.deg2rad(10.0)
MAX_SAFE_RESET_ATTEMPTS = 100


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
        table_full_size=(1.0, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.82),
        randomize_hole_position=RANDOMIZE_HOLE_POSITION,
        hole_position_radius=HOLE_POSITION_RADIUS,
        hole_yaw_range=HOLE_YAW_RANGE,
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
        self.randomize_hole_position = bool(randomize_hole_position)
        self.hole_position_radius = float(hole_position_radius)
        self.hole_yaw_range = float(hole_yaw_range)
        if self.hole_position_radius < 0.0:
            raise ValueError("hole_position_radius must be non-negative")
        if self.hole_yaw_range < 0.0:
            raise ValueError("hole_yaw_range must be non-negative")
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
        arena.set_camera(
            camera_name="agentview",
            pos=AGENTVIEW_POSITION,
            quat=AGENTVIEW_QUATERNION,
        )
        self.peg = SquarePegObject(name="peg")
        # Randomize the model pose at reset, but keep the socket welded to the
        # world throughout each episode.
        self.hole = SquareHoleObject(name="hole", movable=False)
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
        self.hole_planar_joints = tuple(self.hole.joints)
        if self.hole_planar_joints:
            raise RuntimeError("PegInsertion hole must remain fixed after reset")
        # Retain these attributes for callers that inspect the environment.
        self.hole_slide_joints = ()
        self.hole_yaw_joint = None
        self.hole_qvel_addrs = ()
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
        actuator_ids = robot._ref_joint_gripper_actuator_indexes["right"]
        actuator_range = self.sim.model.actuator_ctrlrange[actuator_ids]
        actuator_bias = 0.5 * (actuator_range[:, 1] + actuator_range[:, 0])
        actuator_weight = 0.5 * (actuator_range[:, 1] - actuator_range[:, 0])
        robot.gripper["right"].current_action = np.clip(
            (PREGRASP_GRIPPER_QPOS - actuator_bias) / actuator_weight,
            -1.0,
            1.0,
        )
        self.sim.forward()

        right_pos = np.array(self.sim.data.geom_xpos[self.right_finger_geom_id])
        left_pos = np.array(self.sim.data.geom_xpos[self.left_finger_geom_id])
        pad_midpoint = 0.5 * (right_pos + left_pos)
        pad_mat = np.array(self.sim.data.geom_xmat[self.right_finger_geom_id]).reshape(3, 3)
        peg_mat = np.column_stack((pad_mat[:, 2], pad_mat[:, 1], -pad_mat[:, 0]))
        peg_quat_xyzw = T.mat2quat(peg_mat)
        peg_quat_wxyz = T.convert_quat(peg_quat_xyzw, to="wxyz")
        peg_center = pad_midpoint + pad_mat[:, 0] * (PEG_HALF_LENGTH - PEG_GRASP_OVERLAP / 2.0)

        self.sim.data.set_joint_qpos(
            self.peg_joint,
            np.concatenate([peg_center, peg_quat_wxyz]),
        )
        start, end = self.peg_qvel_addr
        self.sim.data.qvel[start:end] = 0.0
        self.sim.forward()

    def _reset_hole_position(self):
        if self.randomize_hole_position and self.hole_position_radius > 0.0:
            radius = self.hole_position_radius * np.sqrt(self.rng.uniform())
            angle = self.rng.uniform(-np.pi, np.pi)
            offset = radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            offset = np.zeros(2, dtype=float)
        yaw = (
            self.rng.uniform(-self.hole_yaw_range, self.hole_yaw_range)
            if self.randomize_hole_position and self.hole_yaw_range > 0.0
            else 0.0
        )
        self.sim.model.body_pos[self.hole_body_id] = np.array(
            [
                FIXED_HOLE_XY[0] + offset[0],
                FIXED_HOLE_XY[1] + offset[1],
                self.table_offset[2],
            ],
            dtype=float,
        )
        self.sim.model.body_quat[self.hole_body_id] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            dtype=float,
        )
        self.sim.forward()

    def _has_initial_peg_clearance_from_fk(self):
        """Check peg-bottom clearance using forward-kinematic body/site poses."""
        peg_body_pos = np.array(self.sim.data.body_xpos[self.peg_body_id])
        peg_body_mat = np.array(self.sim.data.body_xmat[self.peg_body_id]).reshape(3, 3)
        peg_bottom = peg_body_pos + peg_body_mat @ self.sim.model.site_pos[self.peg_bottom_site_id]
        hole_body_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
        hole_body_mat = np.array(self.sim.data.body_xmat[self.hole_body_id]).reshape(3, 3)
        hole_mouth = hole_body_pos + hole_body_mat @ self.sim.model.site_pos[self.hole_mouth_site_id]
        hole_axis = hole_body_mat[:, 2]
        hole_axis = hole_axis / np.linalg.norm(hole_axis)
        return bool(
            np.dot(peg_bottom - hole_mouth, hole_axis) >= INITIAL_CLEARANCE_ABOVE_HOLE
        )

    def _has_initial_robot_joint_limit_contact(self, margin=1e-4):
        robot = self.robots[0]
        joint_ids = robot._ref_joint_indexes
        limited = self.sim.model.jnt_limited[joint_ids].astype(bool)
        if not np.any(limited):
            return False
        qpos = self.sim.data.qpos[robot._ref_joint_pos_indexes]
        ranges = self.sim.model.jnt_range[joint_ids]
        return bool(
            np.any(
                limited
                & (
                    (qpos <= ranges[:, 0] + margin)
                    | (qpos >= ranges[:, 1] - margin)
                )
            )
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_hole_position()
        for attempt in range(MAX_SAFE_RESET_ATTEMPTS):
            if attempt > 0:
                for robot in self.robots:
                    robot.reset(deterministic=self.deterministic_reset, rng=self.rng)
            if self._has_initial_robot_joint_limit_contact():
                continue
            self._set_pregrasp_pose()
            reset_is_safe = self._has_initial_peg_clearance_from_fk()
            if reset_is_safe:
                break
        else:
            raise RuntimeError(
                "Failed to sample a PegInsertion reset away from robot joint limits, with "
                "initial peg bottom at least "
                f"{INITIAL_CLEARANCE_ABOVE_HOLE:.3f}m above the hole mouth after "
                f"{MAX_SAFE_RESET_ATTEMPTS} attempts"
            )
        self._prev_reward_potential = None

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

        xy_score = 1.0 - np.tanh(50.0 * metrics["xy_error"])
        vertical_score = 1.0 - np.clip(metrics["vertical_angle"] / ALIGNMENT_ANGLE, 0.0, 1.0)
        yaw_score = 1.0 - np.clip(metrics["yaw_error"] / ALIGNMENT_ANGLE, 0.0, 1.0)
        soft_alignment = xy_score * np.mean([vertical_score, yaw_score])
        depth_progress = np.clip(metrics["insertion_depth"] / SUCCESS_DEPTH, 0.0, 1.0)
        insertion = alignment + 0.30 * soft_alignment * depth_progress
        return float(approach), float(alignment), float(insertion)

    def _reward_potential(self):
        metrics = self._compute_insertion_metrics()
        approach_gap = max(-metrics["insertion_depth"], 0.0)
        approach_distance = np.linalg.norm([metrics["xy_error"], approach_gap])
        approach_score = 1.0 - np.tanh(10.0 * approach_distance)
        xy_score = 1.0 - np.tanh(50.0 * metrics["xy_error"])
        vertical_score = 1.0 - np.clip(metrics["vertical_angle"] / ALIGNMENT_ANGLE, 0.0, 1.0)
        yaw_score = 1.0 - np.clip(metrics["yaw_error"] / ALIGNMENT_ANGLE, 0.0, 1.0)
        alignment_score = xy_score * np.mean([vertical_score, yaw_score])
        depth_progress = np.clip(metrics["insertion_depth"] / SUCCESS_DEPTH, 0.0, 1.0)
        gated_alignment = alignment_score**2
        bad_insert = depth_progress * (1.0 - alignment_score)
        return float(
            0.20 * approach_score
            + 0.25 * alignment_score
            + 0.55 * gated_alignment * depth_progress
            - 0.25 * bad_insert
        )

    def _compute_reward(self, action=None, update_reward_state=False):
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            potential = self._reward_potential()
            prev_potential = getattr(self, "_prev_reward_potential", None)
            reward = 0.0 if prev_potential is None else potential - prev_potential
            if update_reward_state:
                self._prev_reward_potential = potential
        else:
            reward = 0.0
        return reward if self.reward_scale is None else reward * self.reward_scale

    def reward(self, action=None):
        return self._compute_reward(action=action, update_reward_state=False)

    def _post_action(self, action):
        reward = self._compute_reward(action=action, update_reward_state=True)
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        metrics = self._compute_insertion_metrics()
        return reward, self.done, {
            "success": int(self._check_success()),
            "peg_insertion_metrics": {
                "insertion_depth": metrics["insertion_depth"],
                "xy_error": metrics["xy_error"],
                "vertical_angle": metrics["vertical_angle"],
                "yaw_error": metrics["yaw_error"],
            },
        }

    def _check_success(self):
        metrics = self._compute_insertion_metrics()
        return bool(
            metrics["insertion_depth"] >= SUCCESS_DEPTH
            and metrics["xy_error"] <= SUCCESS_XY_ERROR
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
