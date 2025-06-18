import os
import torch
import numpy as np
from tqdm import tqdm
import imageio

from opt import config_parser
from dataLoader import dataset_dict
from models.tensoRF import TensorVMSplit
from renderer import OctreeRender_trilinear_fast, evaluation, evaluation_path

def create_spiral_poses(center, up, radii, focus_depth, n_frames=120):
    """生成环绕物体的相机轨迹（spiral）"""
    render_poses = []
    for t in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        c = center + radii * np.array([np.cos(t), -np.sin(t), 0.])
        z = normalize(c - focus_depth * np.array([0, 0, -1]))
        x = normalize(np.cross(up, z))
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([x, y, z], 1)
        pose[:3, 3] = c
        render_poses.append(pose)
    return np.stack(render_poses, 0)

def normalize(x):
    return x / np.linalg.norm(x)

def main():
    args = config_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据集
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test)
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train)

    # 加载模型
    ckpt = torch.load(args.ckpt, map_location=device)
    tensorf = TensorVMSplit(device=device, **ckpt['kwargs'])  # 添加 device 参数
    tensorf.load(ckpt)
    tensorf.eval()
    tensorf.to(device)

    renderer = OctreeRender_trilinear_fast

    # 渲染测试集
    if args.render_test:
        print("Rendering test set...")
        evaluation(test_dataset, tensorf, args, renderer, savePath='render_test', N_vis=args.N_vis, white_bg=args.white_bkgd, device=device)

    # 渲染训练集
    if args.render_train:
        print("Rendering train set...")
        evaluation(train_dataset, tensorf, args, renderer, savePath='render_train', N_vis=args.N_vis, white_bg=args.white_bkgd, device=device)

    # 渲染环绕轨迹
    if args.render_path:
        print("Rendering spiral path video...")
        # 轨迹参数可根据实际数据集调整
        center = np.array([0, 0, 0])
        up = np.array([0, -1, 0])
        radii = np.array([1.0, 1.0, 0.0])
        focus_depth = 4.0
        n_frames = 120
        spiral_poses = create_spiral_poses(center, up, radii, focus_depth, n_frames=n_frames)
        evaluation_path(test_dataset, tensorf, spiral_poses, renderer, savePath='render_spiral', N_vis=n_frames, white_bg=args.white_bkgd, device=device)

        # 合成视频
        imgs = []
        for i in range(n_frames):
            img_path = os.path.join('render_spiral', f'{i:03d}.png')
            imgs.append(imageio.imread(img_path))
        imageio.mimwrite(os.path.join('render_spiral', 'spiral.mp4'), imgs, fps=30, format='ffmpeg')

if __name__ == '__main__':
    main()
