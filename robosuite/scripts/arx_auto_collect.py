from dataclasses import dataclass
import time
import random
import numpy as np
import robosuite as suite
import h5py
import cv2
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R

@dataclass
class AutoCollectConfig:
    robot: str = "Arx5"
    env_name: str = "Lift"
    has_renderer: bool = True
    ignore_done: bool = True
    use_camera_obs: bool = True  # 启用相机观测
    control_freq: int = 20
    gripper_type: str = "ArxGripper"
    record_freq: int = 10  # 数据记录频率 10Hz

class DataRecorder:
    """数据记录器，记录演示数据到HDF5文件"""
    
    def __init__(self, save_dir="demonstrations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取下一个序号
        self.demo_counter = self._get_next_demo_number()
        
        # 临时存储当前演示的数据
        self.current_demo_data = {
            'external_cam': [],
            'robot0_right_eye_in_hand': [],
            'joint_positions': [],
            'ee_pose': [],  # [x, y, z, yaw, pitch, roll]
            'gripper_state': [],
            'timestamps': []
        }
        
        # 视频写入器
        self.video_writer = None
        self.video_path = None
        
        # 记录频率控制
        self.record_interval = 1.0 / 10.0  # 10Hz
        self.last_record_time = 0
        
        print(f"✅ 数据记录器初始化完成，保存目录: {save_dir}")
        print(f"📊 下一个演示序号: {self.demo_counter}")
    
    def _get_next_demo_number(self):
        """获取下一个演示序号"""
        existing_numbers = []
        if os.path.exists(self.save_dir):
            for filename in os.listdir(self.save_dir):
                if filename.startswith('demo_') and filename.endswith('.hdf5'):
                    try:
                        # 提取序号，格式: demo_0.hdf5, demo_1.hdf5, etc.
                        num_str = filename[5:-5]  # 移除 'demo_' 和 '.hdf5'
                        # 只处理纯数字的文件名，忽略时间戳格式
                        if num_str.isdigit():
                            existing_numbers.append(int(num_str))
                    except ValueError:
                        continue
        
        # 返回下一个可用的序号
        if existing_numbers:
            return max(existing_numbers) + 1
        else:
            return 0
    
    def start_new_demo(self):
        """开始新的演示记录"""
        # 清空当前数据
        for key in self.current_demo_data:
            self.current_demo_data[key] = []
        
        # 生成有序的文件名
        self.video_path = os.path.join(self.save_dir, f"{self.demo_counter}.mp4")
        self.hdf5_path = os.path.join(self.save_dir, f"{self.demo_counter}.hdf5")
        
        # 重置视频写入器
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None
        
        # 重置时间
        self.last_record_time = 0
        
        print(f"📹 开始演示 {self.demo_counter} 记录")
    
    def should_record(self, current_time):
        """检查是否应该记录数据（10Hz频率控制）"""
        return (current_time - self.last_record_time) >= self.record_interval
    
    def rotation_matrix_to_euler(self, rotation_matrix):
        """将旋转矩阵转换为欧拉角（yaw, pitch, roll）"""
        r = R.from_matrix(rotation_matrix)
        # 使用 'xyz' 外旋顺序，对应 yaw(z), pitch(y), roll(x)
        euler_angles = r.as_euler('xyz', degrees=False)
        return euler_angles  # [roll, pitch, yaw]
    
    def record_frame(self, env, obs, current_time):
        """记录一帧数据"""
        if not self.should_record(current_time):
            return
        
        try:
            # 获取相机图像
            external_cam_img = obs.get('external_cam_image', None)
            eye_in_hand_img = obs.get('robot0_right_eye_in_hand_image', None)
            
            if external_cam_img is None or eye_in_hand_img is None:
                print("⚠️ 相机图像未找到，跳过记录")
                return
            
            # # 修复图像方向：垂直翻转（MuJoCo渲染的图像通常是倒的）
            # external_cam_img = np.flipud(external_cam_img)
            # eye_in_hand_img = np.flipud(eye_in_hand_img)
            
            # 记录相机图像
            self.current_demo_data['external_cam'].append(external_cam_img)
            self.current_demo_data['robot0_right_eye_in_hand'].append(eye_in_hand_img)
            
            # 获取机器人状态
            robot = env.robots[0]
            
            # 关节位置
            joint_positions = []
            for joint_name in robot.robot_joints:
                joint_id = env.sim.model.joint_name2id(joint_name)
                qpos_addr = env.sim.model.jnt_qposadr[joint_id]
                joint_positions.append(env.sim.data.qpos[qpos_addr])
            self.current_demo_data['joint_positions'].append(np.array(joint_positions))
            
            # 末端执行器位姿
            eef_site_id = robot.eef_site_id["right"]
            ee_pos = env.sim.data.site_xpos[eef_site_id].copy()
            ee_rotation_matrix = env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
            
            # 转换为欧拉角
            euler_angles = self.rotation_matrix_to_euler(ee_rotation_matrix)
            ee_pose = np.concatenate([ee_pos, euler_angles])  # [x, y, z, roll, pitch, yaw]
            self.current_demo_data['ee_pose'].append(ee_pose)
            
            # 夹爪状态（修复为正确的访问方式）
            gripper_joint_name = robot.gripper["right"].joints[0]
            gripper_joint_id = env.sim.model.joint_name2id(gripper_joint_name)
            gripper_qpos_addr = env.sim.model.jnt_qposadr[gripper_joint_id]
            gripper_qpos = env.sim.data.qpos[gripper_qpos_addr]
            self.current_demo_data['gripper_state'].append(gripper_qpos)
            
            # 时间戳
            self.current_demo_data['timestamps'].append(current_time)
            
            # 初始化视频写入器
            if self.video_writer is None and external_cam_img is not None:
                height, width = external_cam_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (width, height))
            
            # 写入视频帧
            if self.video_writer is not None:
                # 转换BGR格式（OpenCV使用BGR）
                if len(external_cam_img.shape) == 3:
                    frame = cv2.cvtColor(external_cam_img, cv2.COLOR_RGB2BGR)
                else:
                    frame = external_cam_img
                self.video_writer.write(frame)
            
            self.last_record_time = current_time
            
        except Exception as e:
            print(f"❌ 记录数据时出错: {e}")
    
    def save_success_demo(self):
        """保存成功的演示数据到HDF5文件"""
        try:
            if not self.current_demo_data['timestamps']:
                print("⚠️ 没有数据可保存")
                return False
            
            with h5py.File(self.hdf5_path, 'w') as f:
                # 保存相机数据
                f.create_dataset('external_cam', data=np.array(self.current_demo_data['external_cam']))
                f.create_dataset('robot0_right_eye_in_hand', data=np.array(self.current_demo_data['robot0_right_eye_in_hand']))
                
                # 保存机器人状态数据
                f.create_dataset('joint_positions', data=np.array(self.current_demo_data['joint_positions']))
                f.create_dataset('ee_pose', data=np.array(self.current_demo_data['ee_pose']))
                f.create_dataset('gripper_state', data=np.array(self.current_demo_data['gripper_state']))
                f.create_dataset('timestamps', data=np.array(self.current_demo_data['timestamps']))
                
                # 元数据
                f.attrs['record_freq'] = 10
                f.attrs['total_frames'] = len(self.current_demo_data['timestamps'])
                f.attrs['duration'] = self.current_demo_data['timestamps'][-1] - self.current_demo_data['timestamps'][0]
                f.attrs['ee_pose_format'] = 'x,y,z,roll,pitch,yaw'
            
            # 关闭视频写入器
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            print(f"💾 成功保存演示数据:")
            print(f"   HDF5: {self.hdf5_path}")
            print(f"   视频: {self.video_path}")
            print(f"   帧数: {len(self.current_demo_data['timestamps'])}")
            print(f"   时长: {self.current_demo_data['timestamps'][-1] - self.current_demo_data['timestamps'][0]:.2f}秒")
            
            # 保存成功后，递增计数器为下一个演示做准备
            self.demo_counter += 1
            return True
            
        except Exception as e:
            print(f"❌ 保存数据时出错: {e}")
            return False
    
    def discard_demo(self):
        """丢弃当前演示数据"""
        # 关闭视频写入器
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # 删除可能已创建的文件
        for file_path in [self.video_path, self.hdf5_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        print("🗑️ 已丢弃失败的演示数据")

class ArxRobotController:
    """ARX5机器人数据收集控制器"""
    
    def __init__(self, env):
        self.env = env
        self.target_reached = False
        self.current_phase = "approach"  # approach -> grasp -> lift
        self.phases = ["approach", "grasp", "lift"]
        self.phase_index = 0
        
        # 控制参数
        self.movement_speed = 0.05
        self.rotation_speed = 0.08  # 增加旋转速度
        self.position_tolerance = 0.01  # 位置容差
        self.orientation_tolerance = 0.25  # 放宽姿态容差，因为姿态控制较慢
        self.grasp_height_offset = 0.3  # 抓取高度偏移
        self.lift_height = 0.4  # 提升高度
        
        # 路径规划
        self.waypoints = []
        self.current_waypoint_index = 0
        
        # 状态跟踪
        self.gripper_closed = False
        
        print("✅ ARX5控制器初始化完成")
    
    def get_ee_position(self):
        """获取末端执行器位置"""
        robot = self.env.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        return self.env.sim.data.site_xpos[eef_site_id].copy()
    
    def get_ee_orientation(self):
        """获取末端执行器姿态（四元数）"""
        robot = self.env.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        # 获取旋转矩阵
        rotation_matrix = self.env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
        # 简化：直接返回旋转矩阵的第一行作为方向向量
        return rotation_matrix[2, :]  # Z轴方向（末端执行器朝向）
    
    def get_cube_position(self):
        """获取方块位置"""
        return self.env.sim.data.body_xpos[self.env.cube_body_id].copy()
    
    def plan_trajectory(self):
        """规划抓取轨迹"""
        cube_pos = self.get_cube_position()
        ee_pos = self.get_ee_position()
        initial_ee_ori = self.get_ee_orientation()  # 获取初始姿态
        
        print(f"🎯 开始规划轨迹:")
        print(f"   当前末端位置: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"   当前末端姿态: [{initial_ee_ori[0]:.3f}, {initial_ee_ori[1]:.3f}, {initial_ee_ori[2]:.3f}]")
        print(f"   方块位置: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        
        # 清空之前的路径点
        self.waypoints = []
        
        # 定义抓取姿态：夹爪朝下
        # 使用方向向量 [0, 0, -1] 表示Z轴朝下
        grasp_orientation = np.array([0.0, 0.0, -1.0])  # 夹爪朝下
        
        # 计算目标位置（方块正上方）
        approach_pos = cube_pos.copy()
        approach_pos[2] += self.grasp_height_offset
        
        # 阶段1a: 先移动到方块正上方，但保持初始姿态（不旋转）
        self.waypoints.append({
            'position': approach_pos,
            'orientation': initial_ee_ori,  # 保持初始姿态
            'gripper': 1.0,  # 打开夹爪
            'phase': 'approach'
        })
        
        # 阶段1b: 在方块正上方调整姿态为朝下
        self.waypoints.append({
            'position': approach_pos,  # 位置不变，停留在方块上方
            'orientation': grasp_orientation,  # 调整为朝下
            'gripper': 1.0,  # 保持打开
            'phase': 'approach'
        })
        
        # 阶段2: 下降到抓取位置
        grasp_pos = cube_pos.copy()
        grasp_pos[2] += 0.15  # 稍微高于方块表面
        self.waypoints.append({
            'position': grasp_pos,
            'orientation': grasp_orientation,
            'gripper': 1.0,  # 保持打开
            'phase': 'grasp'
        })
        
        # 阶段3: 闭合夹爪
        grasp_pos = cube_pos.copy()
        grasp_pos[2] += 0.15 
        self.waypoints.append({
            'position': grasp_pos,
            'orientation': grasp_orientation,
            'gripper': -1.0,  # 闭合夹爪
            'phase': 'grasp'
        })
        
        # 阶段4: 提升方块
        lift_pos = grasp_pos.copy()
        lift_pos[2] += self.lift_height
        self.waypoints.append({
            'position': lift_pos,
            'orientation': grasp_orientation,
            'gripper': -1.0,  # 保持闭合
            'phase': 'lift'
        })
        
        self.current_waypoint_index = 0
        
        print(f"📋 规划了 {len(self.waypoints)} 个路径点:")
        for i, wp in enumerate(self.waypoints):
            pos = wp['position']
            ori = wp['orientation']
            print(f"   {i+1}. 位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                  f"姿态: [{ori[0]:.2f}, {ori[1]:.2f}, {ori[2]:.2f}] "
                  f"夹爪: {wp['gripper']:.1f} 阶段: {wp['phase']}")
        
        return True
    
    def _interpolate_orientation(self, ori_start, ori_end, t):
        """
        在两个方向向量之间进行球面线性插值
        
        Args:
            ori_start: 起始方向向量
            ori_end: 结束方向向量
            t: 插值参数 [0, 1]
        
        Returns:
            插值后的方向向量
        """
        # 归一化输入向量
        ori_start_norm = ori_start / (np.linalg.norm(ori_start) + 1e-8)
        ori_end_norm = ori_end / (np.linalg.norm(ori_end) + 1e-8)
        
        # 计算夹角
        dot = np.clip(np.dot(ori_start_norm, ori_end_norm), -1.0, 1.0)
        theta = np.arccos(dot)
        
        # 如果夹角很小，使用线性插值
        if theta < 1e-6:
            result = (1 - t) * ori_start_norm + t * ori_end_norm
            return result / (np.linalg.norm(result) + 1e-8)
        
        # 球面线性插值 (Slerp)
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        result = w1 * ori_start_norm + w2 * ori_end_norm
        return result / (np.linalg.norm(result) + 1e-8)
    
    def quaternion_distance(self, v1, v2):
        """计算两个方向向量之间的角度距离"""
        # 归一化向量
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        # 计算点积
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        # 返回角度差
        return np.arccos(np.abs(dot_product))
    
    def get_action_to_waypoint(self, target_waypoint):
        """计算到达目标路径点的动作"""
        current_ee_pos = self.get_ee_position()
        current_ee_ori = self.get_ee_orientation()
        target_pos = target_waypoint['position']
        target_ori = target_waypoint['orientation']
        target_gripper = target_waypoint['gripper']
        
        # 计算位置误差
        pos_error = target_pos - current_ee_pos
        pos_distance = np.linalg.norm(pos_error)
        
        # 计算姿态误差
        ori_distance = self.quaternion_distance(current_ee_ori, target_ori)
        
        # 检查是否到达目标（位置和姿态都要满足）
        position_reached = pos_distance < self.position_tolerance
        orientation_reached = ori_distance < self.orientation_tolerance
        
        if position_reached and orientation_reached:
            return None, True  # 返回None表示已到达
        
        # 计算位置运动
        if pos_distance > 0:
            pos_direction = pos_error / pos_distance
            # 使用更激进的运动策略
            if pos_distance > 0.1:  # 如果距离较远，使用最大速度
                pos_movement = pos_direction * self.movement_speed
            else:
                pos_movement = pos_direction * max(0.05, pos_distance * 3)  # 近距离时减速
        else:
            pos_movement = np.zeros(3)
        
        # 计算姿态运动（简化方法）
        if ori_distance > 0:
            # 计算目标方向与当前方向的叉积，得到旋转轴
            current_ori_norm = current_ee_ori / (np.linalg.norm(current_ee_ori) + 1e-8)
            target_ori_norm = target_ori / (np.linalg.norm(target_ori) + 1e-8)
            
            rotation_axis = np.cross(current_ori_norm, target_ori_norm)
            rotation_magnitude = np.linalg.norm(rotation_axis)
            
            if rotation_magnitude > 1e-6:
                # 归一化旋转轴并限制旋转速度
                rotation_axis = rotation_axis / rotation_magnitude
                rotation_speed = min(self.rotation_speed, ori_distance)
                ori_movement = rotation_axis * rotation_speed
            else:
                ori_movement = np.zeros(3)
        else:
            ori_movement = np.zeros(3)
        
        # 构造动作向量
        action_dim = self.env.action_dim
        action = np.zeros(action_dim)
        
        # 位置和姿态控制
        if action_dim >= 6:
            action[:3] = pos_movement  # 位置增量
            action[3:6] = ori_movement  # 姿态增量
        
        # 夹爪控制（ARX5夹爪只需要一个维度）
        if action_dim >= 7:
            action[6] = target_gripper  # 夹爪控制（单一维度控制两个手指）
        
        return action, False
    
    def update(self):
        """更新控制器状态并返回动作"""
        if self.current_waypoint_index >= len(self.waypoints):
            print("🎉 所有路径点执行完成！机器人已到达方块上方")
            return None
        
        current_waypoint = self.waypoints[self.current_waypoint_index]
        action, reached = self.get_action_to_waypoint(current_waypoint)
        
        # 添加超时检测，避免卡死
        if hasattr(self, 'waypoint_start_time'):
            if time.time() - self.waypoint_start_time > 15.0:  # 15秒超时（增加超时时间，因为姿态调整需要更长时间）
                print(f"⚠️  路径点 {self.current_waypoint_index + 1} 超时，强制跳过")
                self.current_waypoint_index += 1
                self.waypoint_start_time = time.time()
                return self.update()
        else:
            self.waypoint_start_time = time.time()
        
        if reached:
            print(f"✅ 到达路径点 {self.current_waypoint_index + 1}/{len(self.waypoints)} "
                  f"({current_waypoint['phase']})")
            
            # 检查夹爪状态是否改变
            if self.current_waypoint_index > 0:
                prev_gripper = self.waypoints[self.current_waypoint_index - 1]['gripper']
                curr_gripper = current_waypoint['gripper']
                if prev_gripper != curr_gripper and curr_gripper < 0:
                    # 夹爪即将闭合，标记需要等待
                    print("🤏 开始夹爪闭合...")
                    self.gripper_wait_time = time.time()
                    self.waiting_for_gripper = True
            
            self.current_waypoint_index += 1
            self.waypoint_start_time = time.time()  # 重置计时器
            
            # 递归调用获取下一个动作
            return self.update()
        
        # 如果正在等待夹爪闭合
        if hasattr(self, 'waiting_for_gripper') and self.waiting_for_gripper:
            elapsed = time.time() - self.gripper_wait_time
            if elapsed < 1.0:  # 等待1秒
                # 继续发送当前动作（保持夹爪闭合命令）
                action_dim = self.env.action_dim
                action = np.zeros(action_dim)
                if action_dim >= 7:
                    action[6] = -1.0  # 持续发送闭合命令
                return action
            else:
                # 等待完成
                print("✅ 夹爪闭合完成")
                self.waiting_for_gripper = False
        
        return action
    
    def is_complete(self):
        """检查是否完成所有任务"""
        return self.current_waypoint_index >= len(self.waypoints)

def create_arx_environment():
    """创建ARX5机器人环境 - 将机器人安装在桌子上"""
    
    # 创建环境 - 启用相机观测
    env = suite.make(
        env_name="Lift",
        robots="Arx5",
        gripper_types="ArxGripper",
        has_renderer=True,
        has_offscreen_renderer=True,  # 启用离屏渲染
        use_camera_obs=True,  # 启用相机观测
        camera_names=["external_cam", "robot0_right_eye_in_hand"],  # 指定要使用的相机
        camera_heights=480,
        camera_widths=640,
        use_object_obs=True,
        control_freq=20,
        horizon=2000,  # 增加时间限制
        reward_shaping=True,
        ignore_done=True,  # 忽略done信号
        hard_reset=True,  # 强制重新加载XML模型
        # 自定义机器人放置
        placement_initializer=None,  # 使用默认放置
    )
    
    print("✅ 环境创建成功")
    print(f"📷 可用相机: {env.camera_names}")
    return env

def collect_demonstration():
    """收集演示数据"""
    # 创建环境
    env = create_arx_environment()
    
    # 创建数据记录器
    recorder = DataRecorder()
    
    # 主循环：持续收集演示
    episode_count = 0
    successful_demos = 0
    
    while True:
        # 重置环境
        obs = env.reset()
        episode_count += 1
        print(f"\n{'='*60}")
        print(f"🔄 第 {episode_count} 次演示开始")
        print(f"{'='*60}")
        
        # 开始新的演示记录
        recorder.start_new_demo()
        demo_start_time = time.time()
        
        # 创建控制器
        controller = ArxRobotController(env)
        
        # 手动调整机器人初始位置 - 让它更接近桌子
        print("🔧 调整机器人初始位置...")
        
        # 设置更好的初始关节角度，让机器人手臂朝向桌子
        robot = env.robots[0]
        joint_angles = [0.0, 0, 0, 0, 0.0, 0.0]  # 让机器人手臂更向前伸展
        
        # 找到机器人关节的qpos索引
        joint_indices = []
        for joint_name in robot.robot_joints:
            joint_id = env.sim.model.joint_name2id(joint_name)
            qpos_addr = env.sim.model.jnt_qposadr[joint_id]
            joint_indices.append(qpos_addr)
        
        # 应用新的关节角度
        for i, angle in enumerate(joint_angles):
            if i < len(joint_indices):
                env.sim.data.qpos[joint_indices[i]] = angle
        
        # 执行前向动力学更新位置
        env.sim.forward()
        
        print("✅ 机器人位置调整完成")
        
        # 等待环境稳定
        print("⏳ 等待环境稳定...")
        for _ in range(100):
            env.step(np.zeros(env.action_dim))
        
        # 规划轨迹
        if not controller.plan_trajectory():
            print("❌ 轨迹规划失败")
            recorder.discard_demo()
            continue
        
        print("\n🚀 开始执行演示...")
        
        step_count = 0
        max_steps_per_episode = 1500  # 每次演示最多1500步
        success_achieved = False
        
        while step_count < max_steps_per_episode:
            # 获取控制动作
            action = controller.update()
            
            if action is None:
                # 如果控制器返回None，使用零动作
                action = np.zeros(env.action_dim)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 记录数据（10Hz频率）
            current_time = time.time() - demo_start_time
            recorder.record_frame(env, obs, current_time)
            
            # 检查任务是否成功
            success = env._check_success()
            if success and not success_achieved:
                print(f"🎉 步骤 {step_count}: 任务成功！")
                success_achieved = True
                # 成功后继续执行一小段时间以稳定状态
                time.sleep(0.5)
                break
            
            # 渲染
            env.render()
            
            step_count += 1
            
            # 打印状态信息
            if step_count % 50 == 0:  # 更频繁地打印状态
                robot = env.robots[0]
                eef_site_id = robot.eef_site_id["right"]
                ee_pos = env.sim.data.site_xpos[eef_site_id]
                cube_pos = env.sim.data.body_xpos[env.cube_body_id]
                
                # 当前路径点信息
                if controller.current_waypoint_index < len(controller.waypoints):
                    current_wp = controller.waypoints[controller.current_waypoint_index]
                    target_pos = current_wp['position']
                    distance = np.linalg.norm(ee_pos - target_pos)
                    print(f"步骤 {step_count}: EE位置 [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] "
                          f"目标 [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] "
                          f"距离: {distance:.3f}m")
                else:
                    print(f"步骤 {step_count}: EE位置 [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] "
                          f"方块位置 [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
            
            # 小延时以便观察
            time.sleep(0.005)  # 减少延时
        
        # 本次演示结束，处理数据
        if success_achieved:
            print(f"✅ 第 {episode_count} 次演示成功完成！（{step_count} 步）")
            if recorder.save_success_demo():
                successful_demos += 1
                print(f"📊 已成功收集 {successful_demos} 个演示")
        else:
            print(f"❌ 第 {episode_count} 次演示失败（超过 {max_steps_per_episode} 步）")
            recorder.discard_demo()
        
        # 短暂等待后开始下一次演示
        time.sleep(1.0)

if __name__ == "__main__":
    collect_demonstration()