import torch
import numpy as np
import os
import sys
from opt import config_parser
from models.tensoRF import TensorVMSplit
from utils import *
from dataLoader.ray_utils import get_rays
from renderer import evaluation, evaluation_path, OctreeRender_trilinear_fast
from dataLoader import dataset_dict

def create_circular_path(radius=3.0, num_views=60):
    """创建水平环绕轨迹的相机位姿"""
    thetas = np.linspace(0, 2*np.pi, num_views)
    c2ws = []
    
    for theta in thetas:
        # 计算相机位置（在水平面上，但稍微抬高一点）
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 1.2  # 将相机位置改为正值，使相机在物体上方
        
        # 创建相机到世界的变换矩阵
        c2w = np.eye(4)
        c2w[:3, 3] = [x, y, z]  # 相机位置
        
        # 让相机始终看向原点，但稍微向下看一点
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        # 设置相机的上方向为z轴正方向（使画面正常）
        up = np.array([0, 0, 1])
        
        # 计算右方向
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # 重新计算上方向，确保三个方向正交
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        
        c2ws.append(c2w)
    
    return np.stack(c2ws)

def main():
    hparams = config_parser()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    dataset = dataset_dict[hparams.dataset_name]
    test_dataset = dataset(hparams.datadir, split='test', downsample=hparams.downsample_test, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = hparams.ndc_ray
    
    # 加载模型
    ckpt = torch.load(hparams.ckpt)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    
    # 从配置文件中获取模型名称
    model_name = hparams.model_name
    tensorf = eval(model_name)(**kwargs)
    tensorf.load(ckpt)
    
    # # 在测试集上评估
    # if hparams.render_test:
    #     print("Rendering test set...")
    #     evaluation(test_dataset, tensorf, hparams, OctreeRender_trilinear_fast, 
    #              os.path.join(hparams.basedir, hparams.expname, 'imgs_test_all'),
    #              N_vis=-1, N_samples=1024, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    
    # 渲染环绕视频
    if hparams.render_path:
        print("Rendering circular path...")
        # 调整相机轨迹参数
        c2ws = create_circular_path(radius=3.0, num_views=60)  # 增加半径到3.0
        
        # 调整渲染参数
        evaluation_path(
            test_dataset=test_dataset,
            tensorf=tensorf,
            c2ws=c2ws,
            renderer=OctreeRender_trilinear_fast,
            savePath=os.path.join(hparams.basedir, hparams.expname, 'imgs_path'),
            N_vis=-1,
            prtx='',
            N_samples=1024,  # 固定采样点数量
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            compute_extra_metrics=True,  # 计算额外指标
            device=device
        )

if __name__ == '__main__':
    main() 