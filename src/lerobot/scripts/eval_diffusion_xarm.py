import threading
import cv2
import numpy as np
import time
import os
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from xarm.wrapper import XArmAPI

# 导入 LeRobot 相关模块
from lerobot.configs import parser
from lerobot.policies.factory import make_policy
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.utils.utils import get_safe_torch_device
from lerobot.envs.utils import preprocess_observation

# 全局共享变量（线程间通信）
latest_base_image = None
latest_wrist_image = None
image_lock = threading.Lock()

# 录制相关变量
recording = False
video_writer = None
recording_lock = threading.Lock()

policy = None
device = None

def start_recording(save_path, fps=30, frame_size=(640, 480)):
    """
    开始录制基础摄像头视频
    
    Args:
        save_path (str): 保存路径（包含文件名，如 "/path/to/video.mp4"）
        fps (int): 帧率，默认30
        frame_size (tuple): 帧大小，默认(640, 480)
    """
    global recording, video_writer
    
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with recording_lock:
        if not recording:
            # 定义编码器和创建VideoWriter对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
            recording = True
            print(f"开始录制视频到: {save_path}")
        else:
            print("警告：已经在录制中")

def stop_recording():
    """停止录制"""
    global recording, video_writer
    
    with recording_lock:
        if recording and video_writer is not None:
            recording = False
            video_writer.release()
            video_writer = None
            print("录制已停止并保存")
        else:
            print("警告：当前没有在录制")

def is_recording():
    """检查是否正在录制"""
    return recording

# 初始化摄像头
def camera_thread(enable_recording=False, save_path="./recorded_video.mp4"):
    """
    摄像头线程
    
    Args:
        enable_recording (bool): 是否启用录制功能
        save_path (str): 录制视频保存路径
    """
    global latest_base_image, latest_wrist_image
    base_camera = cv2.VideoCapture("/dev/video10")
    wrist_camera = cv2.VideoCapture("/dev/video4")

    if not base_camera.isOpened() or not wrist_camera.isOpened():
        print("错误：无法打开摄像头")
        return

    # 获取摄像头实际分辨率
    width = int(base_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(base_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    
    # 如果启用录制，开始录制
    if enable_recording:
        start_recording(save_path, fps=120, frame_size=frame_size)

    try:
        while True:
            ret1, base_img = base_camera.read()
            ret2, wrist_img = wrist_camera.read()
            if not ret1 or not ret2:
                print("警告：图像读取失败")
                continue
                
            with image_lock:
                latest_base_image = base_img.copy()
                latest_wrist_image = wrist_img.copy()

            # 录制基础摄像头画面
            with recording_lock:
                if recording and video_writer is not None:
                    video_writer.write(base_img)

            # 显示图像（可选）
            cv2.imshow("Base Camera", base_img)
            cv2.imshow("Wrist Camera", wrist_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # 按 'r' 键切换录制状态
                if enable_recording:
                    if is_recording():
                        stop_recording()
                    else:
                        # 生成带时间戳的新文件名
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        new_path = save_path.replace('.mp4', f'_{timestamp}.mp4')
                        start_recording(new_path, fps=120, frame_size=frame_size)
                        
    finally:
        # 清理资源
        if is_recording():
            stop_recording()
        base_camera.release()
        wrist_camera.release()
        cv2.destroyAllWindows()

def load_diffusion_policy(pretrained_path: str = None):
    global policy, device

    device = get_safe_torch_device("cuda", log=True)
    policy = DiffusionPolicy.from_pretrained(pretrained_path)
    policy.eval()
    policy.to(device)
    print(f"Diffusion policy loaded from {pretrained_path} and moved to {device}")

def resize_with_pad(image, target_height, target_width):
    h, w = image.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded

def convert_to_uint8(image):
    return image.astype(np.uint8)

# 推理与执行线程
def inference_thread():
    global latest_base_image, latest_wrist_image
    global policy, device

    arm = XArmAPI('192.168.1.222')
    arm.motion_enable(enable=True) 
    arm.set_gripper_enable(enable=True)
    arm.set_mode(6)  
    arm.set_state(0)

    init_qpos = np.array([4.3, 15.5, -9.4, 182.6, 100.4, 0.3])
    arm.set_servo_angle(angle=init_qpos, speed=80, is_radian=False)

    resize_size = 448
    num_steps = 6000
    policy.reset()

    for step in range(num_steps):
        with image_lock:
            if latest_base_image is None or latest_wrist_image is None:
                print("等待图像初始化...")
                time.sleep(0.01)
                continue
            base_img = latest_base_image.copy()
            wrist_img = latest_wrist_image.copy()

        _, joint_rad = arm.get_servo_angle(is_radian=True)
        _, gripper = arm.get_gripper_position()
        state = np.append(np.rad2deg(joint_rad[:-1]), gripper)

        # observation = {
        #     "observation.images.image_1": convert_to_uint8(
        #         resize_with_pad(base_img, resize_size, resize_size)
        #     ),
        #     "observation.images.image_2": convert_to_uint8(
        #         resize_with_pad(wrist_img, resize_size, resize_size)
        #     ),
        #     "agent_pos": state # for preprocess_observation usage
        # }
        observation = {
            "pixels": {
                "image_1": convert_to_uint8(resize_with_pad(base_img, resize_size, resize_size)),
                "image_2": convert_to_uint8(resize_with_pad(wrist_img, resize_size, resize_size)),
            },
            "agent_pos": state
        }

        observation = preprocess_observation(observation)
        # print("Observation keys after preprocess:", observation.keys())
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }
        with torch.inference_mode():
            action = policy.select_action(observation)
            action_np = action.cpu().numpy().squeeze()

            joint_angles = action_np[:-1]
            gripper_position = action_np[-1]
            arm.set_servo_angle(angle=joint_angles, speed=20, is_radian=False)
            arm.set_gripper_position(pos=gripper_position, wait=False)
            print(f"Step {step}, Action: Joint Angles: {joint_angles}, Gripper Position: {gripper_position}")
            time.sleep(1/30)

# 主函数：开启两个线程
if __name__ == "__main__":
    # 模型参数配置
    PRETRAINED_PATH = "/home/bozhao/code/scy/lerobot/outputs/train/2025-08-22/17-01-04_diffusion/checkpoints/500000/pretrained_model"
    # 录制配置
    ENABLE_RECORDING = False  # 设置为True启用录制，False禁用录制
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_PATH = f"/home/bozhao/videos/dp/move_beaker_630/base_camera_{timestamp}.mp4"  # 录制文件保存路径
    
    print(f"录制功能: {'启用' if ENABLE_RECORDING else '禁用'}")
    if ENABLE_RECORDING:
        print(f"录制保存路径: {SAVE_PATH}")
        print("按 'r' 键可以开始/停止录制")
    print("按 'q' 键退出程序")

    load_diffusion_policy(pretrained_path=PRETRAINED_PATH)
    
    cam_thread = threading.Thread(target=camera_thread, args=(ENABLE_RECORDING, SAVE_PATH))
    infer_thread = threading.Thread(target=inference_thread)
    
    cam_thread.start()
    infer_thread.start()
    
    cam_thread.join()
    infer_thread.join()