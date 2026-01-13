from dataclasses import dataclass
import time
import random
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import h5py
import cv2 
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import argparse
import multiprocessing as mp
import logging

# 设置日志级别，避免moviepy输出太多信息
logging.getLogger('moviepy').setLevel(logging.ERROR)

@dataclass
class AutoCollectConfig:
    robot: str = "Arx5"
    env_name: str = "Lift"
    has_renderer: bool = True
    ignore_done: bool = True
    use_camera_obs: bool = True 
    control_freq: int = 20
    gripper_type: str = "ArxGripper"
    record_freq: int = 10 
    save_dir: str = "demonstrations"
    img_size: tuple = (640, 480)  # 原始渲染尺寸
    save_size: tuple = (350, 350) # 保存的目标尺寸
    action: str = "absolute" 
    
Config = AutoCollectConfig()

class DataRecorder:
    def __init__(self):
        self.save_dir = Config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.current_demo_data = {
            'external_cam': [],
            'robot0_right_eye_in_hand': [],
            'joint_states': [],
            'gripper_states': [],
            'actions': [],
            '_timestamps': [] 
        }
        
        self.video_frames = []
        self.record_interval = 1.0 / Config.record_freq
        self.last_record_time = 0
    
    def start_new_demo(self):
        """清空缓存，准备开始新的一集录制"""
        for key in self.current_demo_data:
            self.current_demo_data[key] = []
        self.video_frames = []
        self.last_record_time = 0
    
    def should_record(self, current_time):
        return (current_time - self.last_record_time) >= self.record_interval

    def record_frame(self, env, obs, current_time, action=None):
        if not self.should_record(current_time):
            return
        
        try:
            # 获取原始图像并翻转
            raw_ext_img = obs.get('external_cam_image', None)
            raw_hand_img = obs.get('robot0_right_eye_in_hand_image', None)
            
            if raw_ext_img is None or raw_hand_img is None:
                return

            raw_ext_img = np.flipud(raw_ext_img)
            raw_hand_img = np.flipud(raw_hand_img)
            
            # --- [修改点] 图片裁剪与处理 ---
            target_h, target_w = Config.save_size # 350, 350
            
            # (A) External Camera: 右上角向左偏移20像素，再裁剪350x350
            orig_h, orig_w, _ = raw_ext_img.shape
            
            right_margin = 30 # 向左偏移的像素数
            
            # 检查图像尺寸是否足够大：需要宽度 >= 350 + 20 = 370
            if orig_h >= target_h and orig_w >= (target_w + right_margin):
                crop_y_start = 0
                crop_y_end = target_h
                
                # 计算X轴裁剪范围
                # 终点：宽度 - 20
                crop_x_end = orig_w - right_margin 
                # 起点：终点 - 350
                crop_x_start = crop_x_end - target_w 
                
                ext_img_processed = raw_ext_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            else:
                # 兜底逻辑：如果图片太小不够裁，直接Resize
                ext_img_processed = cv2.resize(raw_ext_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # (B) Eye-in-Hand Camera: Resize
            hand_img_processed = cv2.resize(raw_hand_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            # 存入HDF5缓存
            self.current_demo_data['external_cam'].append(ext_img_processed)
            self.current_demo_data['robot0_right_eye_in_hand'].append(hand_img_processed)
            
            # 存入视频缓存
            self.video_frames.append(ext_img_processed)

            # 记录机器人状态
            robot = env.robots[0]
            joint_positions = []
            for joint_name in robot.robot_joints:
                joint_id = env.sim.model.joint_name2id(joint_name)
                qpos_addr = env.sim.model.jnt_qposadr[joint_id]
                joint_positions.append(env.sim.data.qpos[qpos_addr])
            current_joint_positions = np.array(joint_positions)
            self.current_demo_data['joint_states'].append(current_joint_positions)
            
            gripper_joint_name = robot.gripper["right"].joints[0]
            gripper_joint_id = env.sim.model.joint_name2id(gripper_joint_name)
            gripper_qpos_addr = env.sim.model.jnt_qposadr[gripper_joint_id]
            gripper_qpos = env.sim.data.qpos[gripper_qpos_addr]
            self.current_demo_data['gripper_states'].append(np.array([gripper_qpos, gripper_qpos]))
            
            # 记录动作
            if action is not None:
                joint_increments = action[:6] * 0.1
                if Config.action == "absolute": target_joint_positions = current_joint_positions + joint_increments
                elif Config.action == "relative": target_joint_positions = joint_increments
                else: raise ValueError(f"Unknown action type: {Config.action}")
                gripper_target = action[6]
                target_action = np.append(target_joint_positions, gripper_target)
                self.current_demo_data['actions'].append(target_action)
            else:
                gripper_target = gripper_qpos
                if Config.action == "absolute": target_action = np.append(current_joint_positions, gripper_target)
                elif Config.action == "relative": target_action = np.append(np.zeros(6), gripper_target)
                else: raise ValueError(f"Unknown action type: {Config.action}")
                self.current_demo_data['actions'].append(target_action)
            
            self.current_demo_data['_timestamps'].append(current_time)
            self.last_record_time = current_time
            
        except Exception as e:
            print(f"Error in record_frame: {e}")

    def save_success_demo(self, demo_index):
        """
        保存成功的演示数据 (HDF5, 图片, 视频)
        Args:
            demo_index (int): 演示的全局唯一编号
        """
        try:
            if not self.current_demo_data['_timestamps']:
                return False
            
            # --- 1. 动态构造所有文件路径 ---
            hdf5_path = os.path.join(self.save_dir, f"demo_{demo_index}.hdf5")
            img_path = os.path.join(self.save_dir, f"demo_{demo_index}.jpg")
            video_path = os.path.join(self.save_dir, f"demo_{demo_index}.mp4")

            # --- 2. 保存 HDF5 ---
            with h5py.File(hdf5_path, 'w') as f:
                root = f.create_group('root')
                actions_data = np.array(self.current_demo_data['actions'])
                root.create_dataset('actions', data=actions_data)
                extra_states_group = root.create_group('extra_states')
                
                joint_data = np.array(self.current_demo_data['joint_states'])
                extra_states_group.create_dataset('joint_states', data=joint_data)
                
                gripper_data = np.array(self.current_demo_data['gripper_states'])
                extra_states_group.create_dataset('gripper_states', data=gripper_data)

                view_map = {
                    'external_cam': 'agentview',
                    'robot0_right_eye_in_hand': 'eye_in_hand'
                }

                for original_view_name, target_view_name in view_map.items():
                    image_list = self.current_demo_data[original_view_name]
                    if not image_list: continue
                    
                    images_np = np.array(image_list)
                    images_np_t_c_h_w = np.transpose(images_np, (0, 3, 1, 2))
                    images_np_final = np.expand_dims(images_np_t_c_h_w, axis=0)
                    
                    view_group = root.create_group(target_view_name)
                    view_group.create_dataset('video', data=images_np_final, dtype='u1')

            # --- 3. 保存第一帧图片用于检查 ---
            # if self.current_demo_data['external_cam']:
            #     # 取索引 [0] 即第一帧
            #     check_img = self.current_demo_data['external_cam'][0]
                # Robosuite 是 RGB，OpenCV 需要 BGR
                # check_img_bgr = cv2.cvtColor(check_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(img_path, check_img_bgr)

            # --- 4. 保存视频 (MP4) ---
            if self.video_frames:
                try:
                    import moviepy.editor as mpy
                    # 计算实际录制帧率
                    record_fps = 1.0 / self.record_interval 
                    clip = mpy.ImageSequenceClip(self.video_frames, fps=record_fps)
                    # 写入视频文件
                    clip.write_videofile(
                        video_path, 
                        codec='libx264',
                        audio=False,
                        verbose=False,
                        logger=None,
                        threads=4
                    )
                    clip.close()
                except ImportError:
                    print("Error: moviepy not installed, skipping video save.")
                except Exception as e_vid:
                    print(f"Error saving video {video_path}: {e_vid}")

            return True
            
        except Exception as e:
            print(f"Error save data: {e}")
            for path in [hdf5_path, img_path, video_path]:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
            return False

    def discard_demo(self):
        self.start_new_demo()

class ArxRobotController:
    def __init__(self, env):
        self.env = env
        self.target_reached = False
        self.current_phase = "approach"
        self.phases = ["approach", "grasp", "lift"]
        
        self.movement_speed = 0.2
        self.rotation_speed = 0.1
        self.position_tolerance = 0.01
        self.orientation_tolerance = 0.25
        self.grasp_height_offset = 0.3
        self.lift_height = 0.4
        
        self.waypoints = []
        self.current_waypoint_index = 0
        self.gripper_closed = False
    
    def get_ee_position(self):
        robot = self.env.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        return self.env.sim.data.site_xpos[eef_site_id].copy()
    
    def get_ee_orientation(self):
        robot = self.env.robots[0]
        eef_site_id = robot.eef_site_id["right"]
        rotation_matrix = self.env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
        return rotation_matrix[2, :]
    
    def get_cube_position(self):
        cube_position = self.env.sim.data.body_xpos[self.env.cube_body_id].copy()
        return cube_position
    
    def plan_trajectory(self):
        cube_pos = self.get_cube_position()
        ee_pos = self.get_ee_position()
        initial_ee_ori = self.get_ee_orientation()
        
        self.waypoints = []
        grasp_orientation = np.array([0.0, 0.0, -1.0])
        
        approach_pos = cube_pos.copy()
        approach_pos[2] += self.grasp_height_offset
        
        self.waypoints.append({
            'position': approach_pos,
            'orientation': initial_ee_ori,
            'gripper': 1.0,
            'phase': 'approach'
        })
        
        self.waypoints.append({
            'position': approach_pos,
            'orientation': grasp_orientation,
            'gripper': 1.0,
            'phase': 'approach'
        })
        
        grasp_pos = cube_pos.copy()
        grasp_pos[2] += 0.15
        self.waypoints.append({
            'position': grasp_pos,
            'orientation': grasp_orientation,
            'gripper': 1.0,
            'phase': 'grasp'
        })
        
        grasp_pos = cube_pos.copy()
        grasp_pos[2] += 0.15 
        self.waypoints.append({
            'position': grasp_pos,
            'orientation': grasp_orientation,
            'gripper': -1.0,
            'phase': 'grasp'
        })
        
        lift_pos = grasp_pos.copy()
        lift_pos[2] += self.lift_height
        self.waypoints.append({
            'position': lift_pos,
            'orientation': grasp_orientation,
            'gripper': -1.0,
            'phase': 'lift'
        })
        
        self.current_waypoint_index = 0
        return True
    
    def quaternion_distance(self, v1, v2):
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(np.abs(dot_product))
    
    def get_action_to_waypoint(self, target_waypoint):
        current_ee_pos = self.get_ee_position()
        current_ee_ori = self.get_ee_orientation()
        target_pos = target_waypoint['position']
        target_ori = target_waypoint['orientation']
        target_gripper = target_waypoint['gripper']
        
        pos_error = target_pos - current_ee_pos
        pos_distance = np.linalg.norm(pos_error)
        ori_distance = self.quaternion_distance(current_ee_ori, target_ori)
        
        position_reached = pos_distance < self.position_tolerance
        orientation_reached = ori_distance < self.orientation_tolerance
        
        if position_reached and orientation_reached:
            return None, True
        
        if pos_distance > 0:
            pos_direction = pos_error / pos_distance
            if pos_distance > 0.1:
                pos_movement = pos_direction * self.movement_speed
            else:
                pos_movement = pos_direction * max(0.05, pos_distance * 3)
        else:
            pos_movement = np.zeros(3)
        
        if ori_distance > 0:
            current_ori_norm = current_ee_ori / (np.linalg.norm(current_ee_ori) + 1e-8)
            target_ori_norm = target_ori / (np.linalg.norm(target_ori) + 1e-8)
            rotation_axis = np.cross(current_ori_norm, target_ori_norm)
            rotation_magnitude = np.linalg.norm(rotation_axis)
            
            if rotation_magnitude > 1e-6:
                rotation_axis = rotation_axis / rotation_magnitude
                rotation_speed = min(self.rotation_speed, ori_distance)
                ori_movement = rotation_axis * rotation_speed
            else:
                ori_movement = np.zeros(3)
        else:
            ori_movement = np.zeros(3)
        
        action_dim = self.env.action_dim
        action = np.zeros(action_dim)
        
        if action_dim >= 6:
            action[:3] = pos_movement
            action[3:6] = ori_movement
        
        if action_dim >= 7:
            action[6] = target_gripper
        
        return action, False
    
    def update(self):
        if self.current_waypoint_index >= len(self.waypoints):
            return None
        
        current_waypoint = self.waypoints[self.current_waypoint_index]
        action, reached = self.get_action_to_waypoint(current_waypoint)
        
        if hasattr(self, 'waypoint_start_time'):
            if time.time() - self.waypoint_start_time > 15.0:
                self.current_waypoint_index += 1
                self.waypoint_start_time = time.time()
                return self.update()
        else:
            self.waypoint_start_time = time.time()
        
        if reached:
            if self.current_waypoint_index > 0:
                prev_gripper = self.waypoints[self.current_waypoint_index - 1]['gripper']
                curr_gripper = current_waypoint['gripper']
                if prev_gripper != curr_gripper and curr_gripper < 0:
                    self.gripper_wait_time = time.time()
                    self.waiting_for_gripper = True
            
            self.current_waypoint_index += 1
            self.waypoint_start_time = time.time()
            return self.update()
        
        if hasattr(self, 'waiting_for_gripper') and self.waiting_for_gripper:
            elapsed = time.time() - self.gripper_wait_time
            if elapsed < 1.0:
                action_dim = self.env.action_dim
                action = np.zeros(action_dim)
                if action_dim >= 7:
                    action[6] = -1.0
                return action
            else:
                self.waiting_for_gripper = False
        
        return action

def create_arx_environment(headless=True):
    controller_configs = load_composite_controller_config(robot="arx5")
    
    if controller_configs and "body_parts" in controller_configs:
        gripper_keys = [k for k in controller_configs["body_parts"].keys() if "gripper" in k]
        for g_key in gripper_keys:
            gripper_config = controller_configs["body_parts"][g_key]
            gripper_config["type"] = "JOINT_POSITION"
            gripper_config["input_type"] = "binary"
            gripper_config["kp"] = 1000 
            gripper_config["damping"] = 1

    env = suite.make(
        env_name="Lift",
        robots="Arx5",
        gripper_types="ArxGripper",
        controller_configs=controller_configs,
        has_renderer=(not headless),
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["external_cam", "robot0_right_eye_in_hand"],
        camera_heights=480,
        camera_widths=640,
        use_object_obs=True,
        control_freq=20,
        horizon=2000,
        reward_shaping=True,
        ignore_done=True,
        hard_reset=True,
        placement_initializer=None,
    )
    return env

def worker_collect(worker_id, shared_counter, lock, target_demos, headless):
    env = create_arx_environment(headless=headless)
    recorder = DataRecorder()
    
    pid = os.getpid()
    print(f"[Worker {worker_id}] (PID: {pid}) 启动...")

    while True:
        with lock:
            if shared_counter.value >= target_demos:
                break
        
        obs = env.reset()
        recorder.start_new_demo()
        demo_start_time = time.time()
        controller = ArxRobotController(env)
        
        # 初始化机器人位置
        robot = env.robots[0]
        joint_angles = [0.0, 0, 0, 0, 0.0, 0.0]
        joint_indices = []
        for joint_name in robot.robot_joints:
            joint_id = env.sim.model.joint_name2id(joint_name)
            qpos_addr = env.sim.model.jnt_qposadr[joint_id]
            joint_indices.append(qpos_addr)
        for i, angle in enumerate(joint_angles):
            if i < len(joint_indices):
                env.sim.data.qpos[joint_indices[i]] = angle
        env.sim.forward()
        
        # 等待稳定
        for _ in range(50):
            env.step(np.zeros(env.action_dim))
        
        if not controller.plan_trajectory():
            recorder.discard_demo()
            continue
            
        step_count = 0
        max_steps_per_episode = 1000
        success_achieved = False
        
        while step_count < max_steps_per_episode:
            action = controller.update()
            action_to_step = action if action is not None else np.zeros(env.action_dim)
            obs, reward, done, info = env.step(action_to_step)
            
            current_time = time.time() - demo_start_time
            recorder.record_frame(env, obs, current_time, action)
            
            success = env._check_success()
            if success and not success_achieved:
                success_achieved = True
                break
            
            step_count += 1
        
        if success_achieved:
            with lock:
                if shared_counter.value < target_demos:
                    current_idx = shared_counter.value
                    shared_counter.value += 1
                    
                    # 保存数据
                    saved = recorder.save_success_demo(demo_index=current_idx)
                    
                    if saved:
                        print(f"[Worker {worker_id}] ✅ 已保存 demo_{current_idx} (H5/JPG/MP4) (进度: {shared_counter.value}/{target_demos})")
                else:
                    recorder.discard_demo()
        else:
            recorder.discard_demo()

    print(f"[Worker {worker_id}] 结束任务.")
    env.close()

def run_parallel_collection(num_workers, target_demos, headless):
    manager = mp.Manager()
    shared_counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    processes = []
    
    print(f"🚀 开始多进程数据收集")
    print(f"   目标数量: {target_demos}")
    print(f"   Worker数量: {num_workers}")
    print(f"   保存路径: {Config.save_dir}")
    
    for i in range(num_workers):
        p = mp.Process(
            target=worker_collect,
            args=(i, shared_counter, lock, target_demos, headless)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print(f"\n🎉 所有任务完成！共收集 {shared_counter.value} 个演示。")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=False, help="运行在无头模式")
    parser.add_argument("--num_demos", type=int, default=200, help="需要收集的成功演示数量")
    # default_workers = max(1, os.cpu_count() - 2)
    parser.add_argument("--workers", type=int, default=1, help="并行进程数量")
    
    args = parser.parse_args()

    os.makedirs(Config.save_dir, exist_ok=True)

    run_parallel_collection(
        num_workers=args.workers,
        target_demos=args.num_demos,
        headless=args.headless
    )