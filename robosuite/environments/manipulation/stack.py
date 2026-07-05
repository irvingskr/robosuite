from collections import OrderedDict
from enum import IntEnum

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

REWARD_SHAPING_GAMMA = 0.99
STACK_SUCCESS_REWARD = 2.0
STACK_GRASP_ACQUIRED_REWARD = 0.35
STACK_GRASP_LOST_PENALTY = 0.45
STACK_WRONG_OBJECT_GRASP_PENALTY = 0.50
STACK_GREEN_SHIFT_DEAD_BAND = 0.01
STACK_GREEN_SHIFT_PENALTY_SCALE = 2.5
STACK_GREEN_SHIFT_PENALTY_MAX = 0.25
STACK_REACH_DISTANCE_SCALE = 5.0
STACK_REACH_WEIGHT = 0.30
STACK_CONTACT_WEIGHT = 0.20
STACK_LIFT_START_HEIGHT = 0.02
STACK_LIFT_PROGRESS_HEIGHT = 0.10
STACK_LIFT_COMPLETE_HEIGHT = 0.12
STACK_ALIGNMENT_MIN_HEIGHT = 0.10
STACK_ALIGNMENT_DISTANCE_SCALE = 10.0
STACK_ALIGNMENT_COMPLETE_DISTANCE = 0.035
STACK_ALIGNMENT_EXIT_DISTANCE = 0.055
STACK_PLACEMENT_HEIGHT = 0.10
STACK_LIFT_WEIGHT = 0.45
STACK_ALIGNMENT_WEIGHT = 0.40
STACK_PLACEMENT_WEIGHT = 0.35


class StackRewardStage(IntEnum):
    """Ordered physical milestones used by Stack's augmented-state potential."""

    APPROACH = 0
    LIFT = 1
    ALIGN = 2
    PLACE = 3


class Stack(ManipulationEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
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
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
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

    def reward(self, action):
        """
        Reward function for the task.

        The un-normalized potential is defined over an ordered reward stage and
        the current simulator state. Its nonterminal ranges are [0, 0.50] for
        approaching / contacting the red cube, [0.50, 0.95] for lifting it,
        [0.95, 1.35] for alignment, and [1.35, 1.70] for placement. A later
        range opens only after its physical prerequisite is completed. Success
        has potential 2.0.

        Stack success also provides the sparse un-normalized reward 2.0. Reward
        shaping adds the PBRS term 0.99 * Phi(s') - Phi(s). Grasp acquisition adds
        a one-time 0.35 event reward, while losing a grasp before success adds an
        invalid-drop penalty of -0.45; release on a successful stack is exempt.
        Grasping the green cube adds -0.50, resets task progress, and cannot open
        a later stage. Moving it away from its reset pose adds a bounded
        transition penalty, while alignment and placement remain anchored to
        that reset pose. Holding either cube provides no repeated event bonus,
        and stationary progressed states retain the PBRS discount cost from
        gamma = 0.99.

        If reward_scale is not None, the complete sparse, PBRS, and event result
        is finally multiplied by reward_scale / 2.0.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        return self._compute_reward(action=action, update_reward_state=False)

    def _reward_potential(self, snapshot=None):
        snapshot = self._reward_snapshot() if snapshot is None else snapshot
        stage = getattr(self, "_reward_stage", StackRewardStage.APPROACH)
        return self._stage_potential(stage, **snapshot)

    def _compute_reward(self, action=None, update_reward_state=False):
        snapshot = self._reward_snapshot()
        success = snapshot["success"]
        sparse_reward = STACK_SUCCESS_REWARD if success else 0.0
        reward = sparse_reward
        if self.reward_shaping:
            previous_stage = getattr(self, "_reward_stage", StackRewardStage.APPROACH)
            stage = self._next_reward_stage(previous_stage, **snapshot)
            potential = self._stage_potential(stage, **snapshot)
            prev_potential = getattr(self, "_prev_reward_potential", None)
            prev_red_grasped = getattr(self, "_prev_red_grasped", None)
            prev_green_grasped = getattr(self, "_prev_green_grasped", None)
            prev_green_shift = getattr(self, "_prev_green_effective_shift", None)
            if prev_potential is not None:
                reward += REWARD_SHAPING_GAMMA * potential - prev_potential
            if prev_red_grasped is not None:
                reward += self._grasp_event_reward(
                    prev_red_grasped,
                    snapshot["red_grasped"],
                    success,
                )
            if prev_green_grasped is not None:
                reward += self._green_grasp_event_reward(
                    prev_green_grasped,
                    snapshot["green_grasped"],
                )
            if prev_green_shift is not None:
                reward += self._green_disturbance_reward(
                    prev_green_shift,
                    snapshot["green_effective_shift"],
                )
            if update_reward_state:
                self._reward_stage = stage
                self._prev_reward_potential = potential
                self._prev_red_grasped = snapshot["red_grasped"]
                self._prev_green_grasped = snapshot["green_grasped"]
                self._prev_green_effective_shift = snapshot["green_effective_shift"]

        if self.reward_scale is not None:
            reward *= self.reward_scale / STACK_SUCCESS_REWARD
        return reward

    def _post_action(self, action):
        reward = self._compute_reward(action=action, update_reward_state=True)
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, {}

    @staticmethod
    def _grasp_event_reward(previous_grasped, grasped, success):
        if grasped and not previous_grasped:
            return STACK_GRASP_ACQUIRED_REWARD
        if previous_grasped and not grasped and not success:
            return -STACK_GRASP_LOST_PENALTY
        return 0.0

    @staticmethod
    def _green_grasp_event_reward(previous_grasped, grasped):
        if grasped and not previous_grasped:
            return -STACK_WRONG_OBJECT_GRASP_PENALTY
        return 0.0

    @staticmethod
    def _green_disturbance_reward(previous_shift, shift):
        increase = max(shift - previous_shift, 0.0)
        return -float(
            min(
                STACK_GREEN_SHIFT_PENALTY_SCALE * increase,
                STACK_GREEN_SHIFT_PENALTY_MAX,
            )
        )

    @staticmethod
    def _target_geometry(cube_a_pos, target_pos):
        horizontal_distance = np.linalg.norm(cube_a_pos[:2] - target_pos[:2])
        target_height_error = abs(cube_a_pos[2] - (target_pos[2] + 0.045))
        return float(horizontal_distance), float(target_height_error)

    @staticmethod
    def _green_effective_shift(cube_b_pos, target_pos):
        shift = np.linalg.norm(cube_b_pos[:2] - target_pos[:2])
        return float(max(shift - STACK_GREEN_SHIFT_DEAD_BAND, 0.0))

    @staticmethod
    def _reach_contact_potential(distance, left_contact, right_contact):
        reach = 1.0 - np.tanh(STACK_REACH_DISTANCE_SCALE * distance)
        contact = 0.5 * (float(left_contact) + float(right_contact))
        return float(STACK_REACH_WEIGHT * reach + STACK_CONTACT_WEIGHT * contact)

    @staticmethod
    def _next_reward_stage(
        previous_stage,
        *,
        red_grasped,
        green_grasped,
        red_height,
        table_height,
        horizontal_distance,
        success,
        **_,
    ):
        previous_stage = StackRewardStage(previous_stage)
        if success:
            return previous_stage
        if green_grasped or not red_grasped:
            return StackRewardStage.APPROACH
        if previous_stage is StackRewardStage.APPROACH:
            return StackRewardStage.LIFT
        if (
            previous_stage is StackRewardStage.LIFT
            and red_height >= table_height + STACK_LIFT_COMPLETE_HEIGHT
        ):
            return StackRewardStage.ALIGN
        if (
            previous_stage is StackRewardStage.ALIGN
            and red_height >= table_height + STACK_ALIGNMENT_MIN_HEIGHT
            and horizontal_distance <= STACK_ALIGNMENT_COMPLETE_DISTANCE
        ):
            return StackRewardStage.PLACE
        if (
            previous_stage is StackRewardStage.PLACE
            and horizontal_distance > STACK_ALIGNMENT_EXIT_DISTANCE
        ):
            return StackRewardStage.ALIGN
        return previous_stage

    @staticmethod
    def _stage_potential(
        stage,
        *,
        distance,
        red_left_contact,
        red_right_contact,
        red_grasped,
        green_grasped,
        red_height,
        table_height,
        horizontal_distance,
        target_height_error,
        success,
        **_,
    ):
        if success:
            return STACK_SUCCESS_REWARD
        if green_grasped:
            return 0.0

        stage = StackRewardStage(stage)
        if stage is StackRewardStage.APPROACH:
            return Stack._reach_contact_potential(
                distance=distance,
                left_contact=red_left_contact,
                right_contact=red_right_contact,
            )
        if not red_grasped:
            return 0.0
        if stage is StackRewardStage.LIFT:
            lift = np.clip(
                (red_height - (table_height + STACK_LIFT_START_HEIGHT))
                / STACK_LIFT_PROGRESS_HEIGHT,
                0.0,
                1.0,
            )
            return float(0.50 + STACK_LIFT_WEIGHT * lift)
        if stage is StackRewardStage.ALIGN:
            if red_height < table_height + STACK_ALIGNMENT_MIN_HEIGHT:
                return 0.95
            alignment = 1.0 - np.tanh(STACK_ALIGNMENT_DISTANCE_SCALE * horizontal_distance)
            return float(0.95 + STACK_ALIGNMENT_WEIGHT * alignment)

        placement = 1.0 - np.clip(target_height_error / STACK_PLACEMENT_HEIGHT, 0.0, 1.0)
        return float(1.35 + STACK_PLACEMENT_WEIGHT * placement)

    def _grasp_contacts(self, obj=None):
        object_geoms = (self.cubeA if obj is None else obj).contact_geoms

        def contacts_for_gripper(gripper):
            left_contact = self.check_contact(
                gripper.important_geoms["left_fingerpad"],
                object_geoms,
            )
            right_contact = self.check_contact(
                gripper.important_geoms["right_fingerpad"],
                object_geoms,
            )
            return bool(left_contact), bool(right_contact)

        grippers = self.robots[0].gripper
        if isinstance(grippers, dict):
            contact_pairs = [contacts_for_gripper(gripper) for gripper in grippers.values()]
            return max(contact_pairs, key=sum, default=(False, False))
        return contacts_for_gripper(grippers)

    def _cube_grasped(self, obj=None):
        if obj is None:
            left_contact, right_contact = self._grasp_contacts()
        else:
            left_contact, right_contact = self._grasp_contacts(obj)
        return left_contact and right_contact

    def _reward_snapshot(self):
        cube_a_pos = np.array(self.sim.data.body_xpos[self.cubeA_body_id], copy=True)
        cube_b_pos = np.array(self.sim.data.body_xpos[self.cubeB_body_id], copy=True)
        target_pos = np.array(
            getattr(self, "_stack_reward_target_pos", cube_b_pos),
            copy=True,
        )
        distance = min(
            np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - cube_a_pos)
            for arm in self.robots[0].arms
        )
        red_left_contact, red_right_contact = self._grasp_contacts()
        green_left_contact, green_right_contact = self._grasp_contacts(self.cubeB)
        red_grasped = red_left_contact and red_right_contact
        green_grasped = green_left_contact and green_right_contact
        horizontal_distance, target_height_error = self._target_geometry(cube_a_pos, target_pos)
        cube_a_lifted = cube_a_pos[2] > self.table_offset[2] + 0.04
        success = bool(
            not red_grasped
            and cube_a_lifted
            and self.check_contact(self.cubeA, self.cubeB)
        )
        return {
            "distance": float(distance),
            "red_left_contact": red_left_contact,
            "red_right_contact": red_right_contact,
            "red_grasped": red_grasped,
            "green_grasped": green_grasped,
            "red_height": float(cube_a_pos[2]),
            "table_height": float(self.table_offset[2]),
            "horizontal_distance": horizontal_distance,
            "target_height_error": target_height_error,
            "green_effective_shift": self._green_effective_shift(cube_b_pos, target_pos),
            "success": success,
        }

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple containing approach potential, active ordered-stage
            potential, and physical stack-success potential.
        """
        snapshot = self._reward_snapshot()
        stage = getattr(self, "_reward_stage", StackRewardStage.APPROACH)
        progress_snapshot = dict(snapshot)
        progress_snapshot["success"] = False
        approach = self._stage_potential(StackRewardStage.APPROACH, **progress_snapshot)
        stage_potential = self._stage_potential(stage, **progress_snapshot)
        success = STACK_SUCCESS_REWARD if snapshot["success"] else 0.0
        return approach, stage_potential, success

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[0.15, 0.30],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.sim.forward()
        self._stack_reward_target_pos = np.array(
            self.sim.data.body_xpos[self.cubeB_body_id],
            copy=True,
        )
        self._reward_stage = StackRewardStage.APPROACH
        if self.reward_shaping:
            snapshot = self._reward_snapshot()
            self._prev_reward_potential = self._stage_potential(
                self._reward_stage,
                **snapshot,
            )
            self._prev_red_grasped = snapshot["red_grasped"]
            self._prev_green_grasped = snapshot["green_grasped"]
            self._prev_green_effective_shift = snapshot["green_effective_shift"]
        else:
            self._prev_reward_potential = None
            self._prev_red_grasped = None
            self._prev_green_grasped = None
            self._prev_green_effective_shift = None

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeA_body_id])

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeB_body_id])

            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache
                    else np.zeros(3)
                )

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors = [cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat, cubeA_to_cubeB]
            sensors += [
                self._get_obj_eef_sensor(full_pf, f"{cube}_pos", f"{arm_pf}gripper_to_{cube}", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                for cube in ["cubeA", "cubeB"]
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        return self._reward_snapshot()["success"]

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)
