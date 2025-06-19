import torch  
import numpy as np  
import os  
from tqdm import tqdm  
import cv2  
from scene import Scene  
from gaussian_renderer import render  
from utils.general_utils import safe_state  
from argparse import ArgumentParser  
from arguments import ModelParams, PipelineParams, get_combined_args  
from gaussian_renderer import GaussianModel  
from scene.cameras import Camera  
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  
from PIL import Image  
  
def create_circular_path(center, radius, height, num_frames=120):  
    """创建环绕物体的圆形相机轨迹"""  
    angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)  
    positions = []  
      
    for angle in angles:  
        x = center[0] + radius * np.cos(angle)  
        y = center[1] + radius * np.sin(angle)  
        z = center[2] + height  
        positions.append([x, y, z])  
      
    return np.array(positions)  
  
def look_at_matrix(eye, target, up):  
    """计算look-at变换矩阵"""  
    forward = target - eye  
    forward = forward / np.linalg.norm(forward)  
      
    right = np.cross(forward, up)  
    right = right / np.linalg.norm(right)  
      
    up = np.cross(right, forward)  
      
    R = np.array([right, up, -forward])  
    T = -R @ eye  
      
    return R, T  
  
def create_camera_from_pose(position, target, up, fov, width, height, uid):  
    """从相机位置和朝向创建Camera对象"""  
    R, T = look_at_matrix(position, target, up)  
      
    # 创建虚拟图像（黑色图像）  
    dummy_image = Image.new('RGB', (width, height), (0, 0, 0))  
      
    camera = Camera(  
        resolution=(width, height),  
        colmap_id=uid,  
        R=R,  
        T=T,  
        FoVx=fov,  
        FoVy=fov,  
        depth_params=None,  
        image=dummy_image,  
        invdepthmap=None,  
        image_name=f"frame_{uid:05d}",  
        uid=uid  
    )  
      
    return camera  
  
def render_video(model_path, output_path="video_output", num_frames=120,   
                radius=3.0, height=0.5, fps=30, resolution=(1920, 1080)):  
    """渲染环绕物体的视频"""  
      
    # 解析参数  
    parser = ArgumentParser(description="Video rendering script")  
    model = ModelParams(parser, sentinel=True)  
    pipeline = PipelineParams(parser)  
      
    # 模拟命令行参数  
    import sys  
    sys.argv = ['render_video.py', '-m', model_path]  
    args = get_combined_args(parser)  
      
    # 初始化系统状态  
    safe_state(False)  
      
    # 加载模型  
    gaussians = GaussianModel(args.sh_degree)  
    scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)  
      
    # 设置背景颜色  
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]  
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  
      
    # 计算场景中心（基于训练相机位置）  
    train_cameras = scene.getTrainCameras()  
    camera_positions = []  
    for cam in train_cameras:  
        # 从world_view_transform矩阵中提取相机位置  
        w2c = cam.world_view_transform.cpu().numpy()  
        c2w = np.linalg.inv(w2c)  
        camera_positions.append(c2w[:3, 3])  
      
    camera_positions = np.array(camera_positions)  
    scene_center = np.mean(camera_positions, axis=0)  
      
    print(f"场景中心: {scene_center}")  
    print(f"将渲染 {num_frames} 帧，半径 {radius}")  
      
    # 创建圆形轨迹  
    camera_path = create_circular_path(scene_center, radius, height, num_frames)  
      
    # 创建输出目录  
    os.makedirs(output_path, exist_ok=True)  
    frames_dir = os.path.join(output_path, "frames")  
    os.makedirs(frames_dir, exist_ok=True)  
      
    # 渲染每一帧  
    rendered_frames = []  
      
    with torch.no_grad():  
        for i, position in enumerate(tqdm(camera_path, desc="渲染进度")):  
            # 创建相机  
            target = scene_center  # 始终看向场景中心  
            up = np.array([0, 0, 1])  # Z轴向上  
            fov = train_cameras[0].FoVx  # 使用训练相机的FOV  
              
            camera = create_camera_from_pose(  
                position, target, up, fov,   
                resolution[0], resolution[1], i  
            )  
              
            # 渲染图像  
            render_result = render(camera, gaussians, pipeline.extract(args), background)  
            rendered_image = render_result["render"]  
              
            # 转换为numpy数组并保存  
            image_np = rendered_image.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()  
            image_np = (image_np * 255).astype(np.uint8)  
              
            # 保存帧  
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")  
            cv2.imwrite(frame_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))  
            rendered_frames.append(image_np)  
      
    # 创建视频  
    video_path = os.path.join(output_path, "rendered_video.mp4")  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, resolution)  
      
    print("正在生成视频...")  
    for frame in tqdm(rendered_frames, desc="写入视频"):  
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  
      
    video_writer.release()  
    print(f"视频已保存到: {video_path}")  
      
    return video_path  
  
# 高级版本：支持更多轨迹类型  
def render_custom_trajectory(model_path, trajectory_type="circular", **kwargs):  
    """支持多种轨迹类型的渲染函数"""  
      
    if trajectory_type == "circular":  
        return render_video(model_path, **kwargs)  
      
    elif trajectory_type == "spiral":  
        # 螺旋轨迹实现  
        def create_spiral_path(center, radius, height_range, num_frames):  
            angles = np.linspace(0, 4*np.pi, num_frames)  # 两圈  
            heights = np.linspace(height_range[0], height_range[1], num_frames)  
            positions = []  
              
            for angle, h in zip(angles, heights):  
                x = center[0] + radius * np.cos(angle)  
                y = center[1] + radius * np.sin(angle)  
                z = center[2] + h  
                positions.append([x, y, z])  
              
            return np.array(positions)  
          
        # 修改轨迹生成部分...  
          
    elif trajectory_type == "figure8":  
        # 8字形轨迹实现  
        def create_figure8_path(center, radius, num_frames):  
            t = np.linspace(0, 2*np.pi, num_frames)  
            positions = []  
              
            for time in t:  
                x = center[0] + radius * np.sin(time)  
                y = center[1] + radius * np.sin(time) * np.cos(time)  
                z = center[2] + 0.5  
                positions.append([x, y, z])  
              
            return np.array(positions)  
  
if __name__ == "__main__":  
    # 使用示例  
    model_path = r"D:\DATA\51\gaussian-splatting\output\d4cc7391-a"  # 替换为您的模型路径  
      
    # 基础圆形轨迹  
    video_path = render_video(  
        model_path=model_path,  
        output_path="video_output",  
        num_frames=120,  
        radius=4.0,  
        height=10,  
        fps=30,  
        resolution=(1920, 1080)  
    )  
      
    print(f"视频渲染完成: {video_path}")