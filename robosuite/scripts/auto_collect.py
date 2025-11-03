from dataclasses import dataclass
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.robots import Panda
from robosuite.models.robots import Arx5
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models import MujocoWorldBase
from robosuite.models.objects import BallObject
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_joint
from robosuite.utils.ik_utils import IKSolver, get_nullspace_gains
import mujoco
import mujoco.viewer
import numpy as np
import pyrallis
import random
import math
import time

@dataclass
class AutoCollectConfig:
    robot: str = "Arx5"
    env_name: str = "LiftOnTable"
    has_renderer: bool = True
    ignore_done: bool = True
    use_camera_obs: bool = False
    control_freq: int = 20

class RobotController:
    """机器人路径规划和控制器"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.target_reached = False
        self.path_points = []
        self.current_path_index = 0
        self.movement_speed = 0.005  # 降低运动速度，更容易观察
        self.position_tolerance = 0.03  # 稍微放宽容差
        
        # 手动控制标志
        self.manual_mode = True
        self.waiting_for_trigger = True
        self.path_planning_done = False
        self.last_key_time = 0
        
        # IKSolver 相关
        self.ik_solver = None
               
        # 找到关键的身体部位和关节索引
        self.setup_indices()
        
        # 初始化 IK 求解器
        self.setup_ik_solver()
        
    def setup_indices(self):
        """设置机器人关节和身体部位的索引"""
        # 找到ARX5关节索引
        self.joint_indices = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and "robot0_joint" in joint_name:
                self.joint_indices.append(i)
        
        # 找到末端执行器(夹具)的身体ID
        self.gripper_body_id = None
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                # 更精确的夹爪身体匹配
                if any(name in body_name for name in ["gripper0_right_gripper", "gripper0_eef", "robot0_right_hand"]):
                    self.gripper_body_id = i
                    print(f"🤏 找到夹爪身体: {body_name} (ID: {i})")
                    break
        
        # 找到方块的身体ID
        self.box_body_id = None
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and "box" in body_name:
                self.box_body_id = i
                break
                
        print(f"找到 {len(self.joint_indices)} 个关节")
        print(f"夹具身体ID: {self.gripper_body_id}")
        print(f"方块身体ID: {self.box_body_id}")
    
    def move_to_initial_position(self):
        """将机器人移动到标准初始位置"""
        print("🏠 正在移动到初始位置...")
        
        # 获取当前夹爪位置
        current_pos = self.get_gripper_position()
        if current_pos is None:
            print("❌ 无法获取当前夹具位置")
            return False
        
        print(f"🎯 当前夹爪位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        
        # 定义更保守的初始位置 - 基于当前位置适当调整
        # 确保在机器人的工作空间内
        initial_position = np.array([
            max(0.5, min(0.7, current_pos[0])),  # X: 限制在合理范围内
            max(-0.1, min(0.1, current_pos[1])), # Y: 接近中央
            max(0.9, current_pos[2])             # Z: 确保不低于当前高度
        ])
        
        print(f"🎯 目标初始位置: [{initial_position[0]:.3f}, {initial_position[1]:.3f}, {initial_position[2]:.3f}]")
        
        # 使用 IK 求解器移动到初始位置
        max_attempts = 30  # 减少尝试次数
        tolerance = 0.08   # 放宽容差
        
        for attempt in range(max_attempts):
            current_pos = self.get_gripper_position()
            if current_pos is None:
                print("❌ 无法获取当前夹具位置")
                return False
            
            # 计算距离
            distance = np.linalg.norm(current_pos - initial_position)
            
            if distance < tolerance:
                print(f"✅ 已到达初始位置！")
                print(f"   目标位置: [{initial_position[0]:.3f}, {initial_position[1]:.3f}, {initial_position[2]:.3f}]")
                print(f"   当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                print(f"   误差: {distance:.3f}m")
                return True
            
            # 计算中间目标（平滑移动），使用更小的步长
            direction = initial_position - current_pos
            step_size = min(0.005, distance * 0.1)  # 更小更保守的步长
            target_pos = current_pos + direction / np.linalg.norm(direction) * step_size
            
            # 使用逆运动学求解
            success = self.inverse_kinematics_simple(target_pos)
            
            # 物理仿真步进 - 更多步数让系统稳定
            for _ in range(20):
                mujoco.mj_step(self.model, self.data)
            
            # 每5次尝试打印一次进度
            if attempt % 5 == 0:
                print(f"   🔄 移动进度 {attempt}/{max_attempts}, 距离: {distance:.3f}m")
        
        # 如果没能到达，检查是否在可接受范围内
        final_pos = self.get_gripper_position()
        final_distance = np.linalg.norm(final_pos - initial_position) if final_pos is not None else float('inf')
        
        # 更宽松的最终检查
        if final_distance < tolerance * 1.5:  # 允许更大的容差
            print(f"⚠️  接近初始位置（距离: {final_distance:.3f}m），继续执行")
            return True
        else:
            print(f"❌ 未能到达初始位置，最终距离: {final_distance:.3f}m")
            print(f"   当前位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
            print("💡 尝试直接从当前位置开始路径规划...")
            return True  # 即使没到达理想位置也继续，让用户决定
    
    def setup_ik_solver(self):
        """初始化 IK 求解器"""
        try:
            # 获取关节名称
            joint_names = []
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and "robot0_joint" in joint_name:
                    joint_names.append(joint_name)
            
            # 寻找末端执行器站点 - 选择最主要的一个
            end_effector_sites = []
            main_site = None
            
            for i in range(self.model.nsite):
                site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                if site_name:
                    # 优先选择 grip_site 或 eef 相关的站点
                    if "grip_site" in site_name and "cylinder" not in site_name:
                        main_site = site_name
                        print(f"🎯 选择主要末端执行器站点: {site_name}")
                        break
                    elif "eef" in site_name and main_site is None:
                        main_site = site_name
                        
            if main_site:
                end_effector_sites = [main_site]
            else:
                # 如果没找到，寻找包含 gripper 的站点
                for i in range(self.model.nsite):
                    site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                    if site_name and "gripper" in site_name:
                        end_effector_sites.append(site_name)
                        print(f"🎯 找到末端执行器站点: {site_name}")
                        break
            
            # 如果没找到标准站点，尝试创建一个基于末端执行器的站点
            if not end_effector_sites:
                # 查找可能的末端执行器身体
                for i in range(self.model.nbody):
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and ("hand" in body_name or "eef" in body_name or "gripper" in body_name):
                        end_effector_sites.append(f"{body_name}_center")
                        print(f"🎯 使用身体中心作为末端执行器: {body_name}_center")
                        break
            
            if not end_effector_sites:
                print("⚠️  未找到末端执行器站点，将使用身体位置作为替代")
                self.ik_solver = None
                return
            
            # 计算零空间增益
            nullspace_weights = {}
            for joint_name in joint_names:
                if "joint1" in joint_name or "joint2" in joint_name:  # 基座和肩部关节
                    nullspace_weights[joint_name] = 1.0
                elif "joint3" in joint_name:  # 肘部关节
                    nullspace_weights[joint_name] = 0.8
                else:  # 腕部关节
                    nullspace_weights[joint_name] = 0.5
            
            Kn = get_nullspace_gains(joint_names, nullspace_weights)
            
            # 机器人配置
            robot_config = {
                "end_effector_sites": end_effector_sites,
                "joint_names": joint_names,
                "mocap_bodies": [],
                "nullspace_gains": Kn,
            }
            
            # 创建 IK 求解器
            self.ik_solver = IKSolver(
                model=self.model,
                data=self.data,
                robot_config=robot_config,
                damping=0.05,  # 适中的阻尼，平衡稳定性和精度
                integration_dt=0.05,  # 较小的时间步，提高精度
                max_dq=1.0,  # 降低最大关节速度，提高稳定性
                input_rotation_repr="axis_angle",
                input_type="keyboard",
                debug=False
            )
            
            print(f"✅ IK求解器初始化成功")
            print(f"   - 控制关节: {joint_names}")
            print(f"   - 末端执行器: {end_effector_sites}")
            print(f"   - 控制维度: {self.ik_solver.control_dim}")
            
        except Exception as e:
            print(f"❌ IK求解器初始化失败: {e}")
            print("   将使用简单的数值方法作为后备")
            self.ik_solver = None
    
    def inverse_kinematics_simple(self, target_pos, target_quat=None, max_attempts=10):
        """
        使用 IKSolver 计算并应用关节控制指令，以到达目标位置。
        
        Args:
            target_pos (np.array): 目标位置 (x, y, z)
            target_quat (np.array, optional): 目标姿态 (w, x, y, z)。如果为 None，则保持当前姿态。
            max_attempts (int): 求解器尝试的次数。

        Returns:
            bool: 是否成功计算并应用了动作。
        """
        
        if self.ik_solver is None:
            print("⚠️  IK 求解器不可用，跳过 IK 步骤。")
            return False
            
        try:
            # 1. 获取当前关节位置作为IK的初始猜测
            q_init = self.data.qpos[self.joint_indices]
            
            # 2. 准备目标姿态
            if target_quat is None:
                # 如果没有提供目标姿态，则尝试保持当前姿态
                # 注意：IKSolver 需要 (x, y, z, qx, qy, qz, qw) 格式
                current_gripper_quat = self.data.xquat[self.gripper_body_id]
                target_orientation = current_gripper_quat[[1, 2, 3, 0]] # 转换为 (qx, qy, qz, qw)
            else:
                target_orientation = target_quat[[1, 2, 3, 0]] # 转换为 (qx, qy, qz, qw)

            # 3. 构造 IK 求解器的目标
            # IKSolver 需要 (x, y, z, qx, qy, qz, qw) 格式
            target = np.concatenate([target_pos, target_orientation])
            
            # 4. 调用 IK 求解器
            # 我们需要的是关节速度 (dq)，而不是位置 (qpos)
            # 这里的 'solve' 方法通常会计算一个 'delta' (即 dq)
            # robosuite 的 IKSolver.solve() 返回的是关节速度 dq
            
            # 注意：robosuite的IKSolver设计是用来计算 *速度* 的，而不是目标 *位置*。
            # 它需要一个 delta_pos 和 delta_ori 作为输入。
            
            # --- 修正：使用 robosuite IKSolver 的正确方式 ---
            
            # a. 获取当前末端执行器位置和姿态
            current_pos = self.get_gripper_position()
            current_quat_xyzw = self.data.xquat[self.gripper_body_id] # (x, y, z, w)
            current_quat_wxyz = current_quat_xyzw[[3, 0, 1, 2]] # (w, x, y, z)
            
            if current_pos is None:
                return False

            # b. 计算位置误差 (delta_pos)
            delta_pos = target_pos - current_pos
            
            # c. 计算姿态误差 (delta_ori)
            # 保持当前姿态，所以姿态误差为0
            # IKSolver 需要一个轴-角(axis-angle)格式的旋转误差
            delta_ori = np.zeros(3) # 保持当前姿态

            # d. 将 delta 组合成求解器需要的格式 (6D: dx, dy, dz, ax, ay, az)
            control_delta = np.concatenate([delta_pos, delta_ori])
            
            # e. 求解关节速度 (dq)
            # 我们限制 delta_pos 的大小，使其更像一个速度指令
            pos_step = 0.1 # 调整这个值来控制移动速度
            delta_pos_norm = np.linalg.norm(delta_pos)
            if delta_pos_norm > pos_step:
                 control_delta[:3] = delta_pos / delta_pos_norm * pos_step

            # 求解关节速度 dq
            #
            # --- 这是修改后的部分 ---
            # 我们将 control_delta 作为第一个*位置*参数传递
            # 并且移除了 is_delta 参数
            #
            dq = self.ik_solver.solve(
                control_delta                    # 传入 6D delta [dx, dy, dz, dax, day, daz]
            )
            # --- 修改结束 ---

            if dq is None:
                print("❌ IK 求解失败")
                return False

            # 5. 将计算出的关节速度 (dq) 应用为控制信号
            # MuJoCo 的执行器 (actuators) 通常期望的是目标位置 (qpos) 或速度 (qvel)
            # 假设你的执行器是位置控制 (position actuators)
            
            # --- 方案A: 如果是位置控制 (qpos) ---
            # 我们需要计算目标 qpos
            
            # 仿真时间步长 (从模型获取)
            dt = self.model.opt.timestep
            
            # 简单的积分： q_target = q_current + dq * dt
            # 注意：这假设 control_freq 和 timestep 一致，在 robosuite 中通常不是这样
            # 在你的代码中，control_freq=20Hz (0.05s)
            control_dt = 1.0 / 20.0 # 你的 control_freq
            
            # 计算目标关节位置
            target_qpos = self.data.qpos[self.joint_indices] + dq * control_dt
            
            # 将目标位置应用到 data.ctrl
            # 确保你的执行器 (actuators) 在 XML 中被正确设置
            # 假设执行器和关节一一对应
            for i, joint_idx in enumerate(self.joint_indices):
                # 找到该关节对应的执行器ID
                # actuator_id = mujoco.mj_modelSensedata(self.model, mujoco.mjtSensor.mjSENS_ACTUATORPOS, joint_idx, -1)
                
                # 在 robosuite 中，执行器通常是按顺序的
                # 一个更稳健的方法是假设前N个执行器对应前N个关节
                # 你的关节有6个
                if i < self.model.nu: # nu 是执行器的数量
                    self.data.ctrl[i] = target_qpos[i] 
                
            return True

            # --- 方案B: 如果是速度控制 (qvel) ---
            # (如果你的执行器是速度控制，使用这个)
            # for i, joint_idx in enumerate(self.joint_indices):
            #     if i < self.model.nu:
            #         self.data.ctrl[i] = dq[i]
            # return True

        except Exception as e:
            print(f"❌ 在 IK 求解或应用时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    #
    # --- 粘贴到这个位置 ---
    #

    def get_gripper_position(self):
        """获取夹具的当前位置"""
        if self.gripper_body_id is not None:
            return self.data.xpos[self.gripper_body_id].copy()
        return None
    
    def get_box_position(self):
        """获取方块的当前位置"""
        if self.box_body_id is not None:
            return self.data.xpos[self.box_body_id].copy()
        return None
    
    def plan_path_to_box(self, approach_height=0.30):
        """规划从初始位置到方块上方的路径"""
        box_pos = self.get_box_position()
        
        if box_pos is None:
            print("无法获取方块位置")
            return False
        
        # 获取当前位置（应该是初始位置）
        current_pos = self.get_gripper_position()
        if current_pos is None:
            print("无法获取夹具位置")
            return False
        
        print(f"规划起始位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        print(f"方块位置: [{box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}]")
        
        # 目标位置：方块正上方
        target_pos = box_pos.copy()
        target_pos[2] += approach_height  # 在方块上方一定高度
        
        print(f"目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # 生成路径点
        self.path_points = []
        
        # 路径点1：从当前位置移动到方块上方的安全高度
        safe_height = max(current_pos[2], target_pos[2], box_pos[2] + 0.3)
        waypoint1 = current_pos.copy()
        waypoint1[2] = safe_height
        self.path_points.append(waypoint1)
        
        # 路径点2：移动到目标XY位置，保持安全高度
        waypoint2 = target_pos.copy()
        waypoint2[2] = safe_height
        self.path_points.append(waypoint2)
        
        # 路径点3：降低到目标高度（方块上方）
        self.path_points.append(target_pos)
        
        # 如果当前位置已经在合适的高度，可以优化路径
        if abs(current_pos[2] - safe_height) < 0.05:
            # 如果已经在安全高度，跳过第一个路径点
            self.path_points = self.path_points[1:]
        
        self.current_path_index = 0
        self.target_reached = False
        
        print(f"规划了 {len(self.path_points)} 个路径点:")
        for i, point in enumerate(self.path_points):
            print(f"  路径点 {i+1}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        
        return True
    
    
    def move_towards_target(self, target_pos):
        """使用简单的关节空间插值向目标移动"""
        gripper_pos = self.get_gripper_position()
        if gripper_pos is None:
            return False
        
        # 计算位置误差
        position_error = target_pos - gripper_pos
        distance = np.linalg.norm(position_error)
        
        if distance < self.position_tolerance:
            return True  # 到达目标
        
        # 计算运动方向
        direction = position_error / distance
        movement = direction * min(self.movement_speed, distance)
        
        # 目标位置
        new_target = gripper_pos + movement
        
        # 尝试逆运动学求解
        success = self.inverse_kinematics_simple(new_target)
        
        return distance < self.position_tolerance
    
    def update_control(self):
        """更新控制器，执行路径跟踪"""
        # 如果是手动模式且在等待触发
        if self.manual_mode and self.waiting_for_trigger:
            return
            
        if self.target_reached or len(self.path_points) == 0:
            return
        
        if self.current_path_index >= len(self.path_points):
            self.target_reached = True
            print("🎉 路径执行完成！机器人已到达方块上方")
            return
        
        # 当前目标点
        current_target = self.path_points[self.current_path_index]
        
        # 向当前目标点移动
        reached = self.move_towards_target(current_target)
        
        if reached:
            print(f"✅ 到达路径点 {self.current_path_index + 1}/{len(self.path_points)}")
            self.current_path_index += 1
            
            # 如果是手动模式，到达一个路径点后暂停
            if self.manual_mode and self.current_path_index < len(self.path_points):
                print(f"📍 等待按 SPACE 键继续到路径点 {self.current_path_index + 1}...")
                self.waiting_for_trigger = True
                return
            
            if self.current_path_index >= len(self.path_points):
                self.target_reached = True
                print("🎯 所有路径点已到达！机器人现在位于方块上方")
    
    def is_waiting(self):
        """检查是否在等待用户输入"""
        return self.manual_mode and self.waiting_for_trigger
    
    def get_status(self):
        """获取控制器状态信息"""
        gripper_pos = self.get_gripper_position()
        box_pos = self.get_box_position()
        
        status = {
            'gripper_position': gripper_pos,
            'box_position': box_pos,
            'current_path_index': self.current_path_index,
            'total_path_points': len(self.path_points),
            'target_reached': self.target_reached
        }
        
        return status

def new_env():
    world = MujocoWorldBase()
    mujoco_arena = TableArena()
    mujoco_arena.set_origin([0.8, 0, 0]) 
    world.merge(mujoco_arena)

    # mujoco_arena.table_offset[2] 存储了桌面相对于其原点的高度 (默认为 0.8)
    table_height = mujoco_arena.table_offset[2] 
    robot_base_pos = [0.45, 0.0, table_height]
    # mujoco_robot = Arx5()
    # gripper = gripper_factory('ArxGripper')
    mujoco_robot = Arx5()
    gripper = gripper_factory('ArxGripper')
    mujoco_robot.add_gripper(gripper)
    mujoco_robot.set_base_xpos(robot_base_pos) 
    world.merge(mujoco_robot)
    
    # 创建随机位置的方块
    # 随机位置: x:[0.6,1.0], y:[-0.35,0.35], z:0.9
    random_x = random.uniform(0.6, 1.0)
    random_y = random.uniform(-0.35, 0.35)
    random_z = 0.9
    
    # 生成随机方向 (四元数格式: w x y z)
    # 绕z轴随机旋转
    random_angle = random.uniform(0, 2 * np.pi)
    quat_w = np.cos(random_angle / 2)
    quat_x = 0
    quat_y = 0  
    quat_z = np.sin(random_angle / 2)
    
    print(f"方块随机位置: x={random_x:.3f}, y={random_y:.3f}, z={random_z}")
    print(f"方块随机角度: {np.degrees(random_angle):.1f}度")
    
    box = BoxObject(
        name="box",
        size=[0.025, 0.025, 0.025],
        rgba=[0, 0, 1, 1]).get_obj()
    
    # random_x = 0.8
    # random_y = 0.0
    # random_z = table_height + 0.025
    box.set('pos', f'{random_x} {random_y} {random_z}')
    # 设置随机方向
    box.set('quat', f'{quat_w} {quat_x} {quat_y} {quat_z}')
    
    world.worldbody.append(box)
    
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    
    # 设置机器人初始姿势 - 防止软塌
    # ARX5机器人的标准站立姿势
    robot_joint_names = []
    print("🤖 检测到的关节:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and "robot0_joint" in joint_name:
            robot_joint_names.append((i, joint_name))
            print(f"  - 关节 {i}: {joint_name}")
    
    # 设置初始关节角度 (单位：弧度)
    initial_joint_angles = {
        "robot0_joint1": 0.0,      # 基座旋转
        "robot0_joint2": -0.3,     # 肩部俯仰 (稍微向前倾)
        "robot0_joint3": 0.5,      # 肘部俯仰 (弯曲)
        "robot0_joint4": 0.0,      # 腕部俯仰
        "robot0_joint5": 0.0,      # 腕部滚转
        "robot0_joint6": 0.0       # 腕部偏转
    }
    
    # 应用初始关节角度
    print("🔧 设置初始关节角度:")
    for joint_idx, joint_name in robot_joint_names:
        if joint_name in initial_joint_angles:
            data.qpos[joint_idx] = initial_joint_angles[joint_name]
            print(f"  ✅ 设置关节 {joint_name} 角度: {initial_joint_angles[joint_name]:.3f} 弧度")
        else:
            print(f"  ❌ 未找到关节 {joint_name} 的初始角度设置")
    
    # 执行前向运动学以更新位置
    mujoco.mj_forward(model, data)
    
    # 创建机器人控制器
    controller = RobotController(model, data)
    
    # 等待几秒让机器人稳定
    print("等待机器人稳定...")
    for _ in range(1000):
        mujoco.mj_step(model, data)
    
    # 移动到初始位置
    print("\n🏠 正在移动到标准初始位置...")
    if not controller.move_to_initial_position():
        print("❌ 无法移动到初始位置，程序退出")
        return
    
    print("✅ 初始位置设置完成！")
    
    # 规划路径
    print("\n🗺️  开始规划路径...")
    if controller.plan_path_to_box(approach_height=0.15):
        print("✅ 路径规划成功！")
        controller.path_planning_done = True
    else:
        print("❌ 路径规划失败！")
        return
    
    print("\n=== 键盘控制说明 ===")
    print("在终端中输入以下命令:")
    print("按回车键: 开始/继续执行下一步")
    print("输入 'q' 然后回车: 切换到自动模式")
    print("输入 'm' 然后回车: 切换回手动模式")
    print("========================")
    
    print("\n⏸️  手动模式启动，按回车键开始执行第一个路径点...")
    print("💡 如果要切换模式，请输入对应字母然后按回车")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45
        
        step = 0
        status_print_interval = 1000  # 减少打印间隔，更频繁检查
        last_waiting_message_time = 0
        
        while viewer.is_running() and data.time < 100:
            # 简单的非阻塞输入检查
            try:
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    user_input = sys.stdin.readline().strip().lower()
                    
                    if user_input == '' or user_input == ' ':  # 回车或空格
                        if controller.is_waiting():
                            controller.waiting_for_trigger = False
                            print("▶️  继续执行...")
                        else:
                            print("💡 当前不需要手动触发")
                    elif user_input == 'q':  # 自动模式
                        controller.manual_mode = False
                        controller.waiting_for_trigger = False
                        print("🤖 切换到自动模式")
                    elif user_input == 'm':  # 手动模式
                        controller.manual_mode = True
                        if not controller.target_reached and controller.current_path_index < len(controller.path_points):
                            controller.waiting_for_trigger = True
                        print("✋ 切换到手动模式")
                        
            except ImportError:
                # 如果select不可用，使用定时自动触发作为后备
                if controller.is_waiting() and step % 5000 == 0 and step > 0:
                    print("⚠️  检测到输入系统不可用，自动继续执行...")
                    controller.waiting_for_trigger = False
            
            # 更新控制器
            controller.update_control()
            
            # 物理仿真步进
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 如果在等待状态，定期提醒
            current_time = time.time()
            if controller.is_waiting() and (current_time - last_waiting_message_time) > 2:
                status = controller.get_status()
                print(f"⏳ 等待中... 当前进度: {status['current_path_index']}/{status['total_path_points']} (按回车继续)")
                last_waiting_message_time = current_time
            
            # 定期打印详细状态
            if step % status_print_interval == 0 and step > 0:
                status = controller.get_status()
                if status['gripper_position'] is not None and status['box_position'] is not None:
                    gripper_pos = status['gripper_position']
                    box_pos = status['box_position']
                    distance = np.linalg.norm(gripper_pos - box_pos)
                    
                    print(f"\n📊 === 步数: {step} ===")
                    print(f"🤖 夹具位置: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
                    print(f"📦 方块位置: [{box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}]")
                    print(f"📏 距离: {distance:.3f}m")
                    print(f"🛤️  路径进度: {status['current_path_index']}/{status['total_path_points']}")
                    print(f"🎯 目标到达: {'是' if status['target_reached'] else '否'}")
                    print(f"⏸️  等待输入: {'是' if controller.is_waiting() else '否'}")
            
            step += 1
            
            # 如果目标已到达，庆祝一下
            if controller.target_reached and step % (status_print_interval * 2) == 0:
                print("\n🎉🎉🎉 机器人已成功到达方块上方！🎉🎉🎉")
                print("💡 可以在此处添加抓取逻辑...")
    
    print("\n程序结束")

if __name__ == "__main__":
    
    cfg = pyrallis.parse(AutoCollectConfig)
    # collect_auto_trajectory(cfg)
    new_env()

