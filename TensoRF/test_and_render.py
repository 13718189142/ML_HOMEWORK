import os
import torch
import numpy as np
from opt import config_parser
from dataLoader import dataset_dict
from models.tensoRF import TensorVMSplit
from renderer import evaluation, evaluation_path, OctreeRender_trilinear_fast
from dataLoader.ray_utils import get_rays
import json
import argparse

def create_circular_path(radius=3.0, num_views=120, height=1.2):
    """创建环绕物体的圆形相机轨迹"""
    thetas = np.linspace(0, 2*np.pi, num_views)
    c2ws = []
    
    for theta in thetas:
        # 计算相机位置
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height
        
        # 创建相机到世界的变换矩阵
        c2w = np.eye(4)
        c2w[:3, 3] = [x, y, z]  # 相机位置
        
        # 让相机始终看向原点
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        # 设置相机的上方向
        up = np.array([0, 0, 1])
        
        # 计算右方向
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # 重新计算上方向，确保正交
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        
        c2ws.append(c2w)
    
    return np.stack(c2ws)

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='NeRF测试和渲染脚本')
    parser.add_argument('--config', type=str, default='configs/lego.txt', 
                        help='配置文件路径')
    parser.add_argument('--ckpt', type=str, 
                        help='模型检查点文件路径(.th文件)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认使用检查点所在目录')
    parser.add_argument('--radius', type=float, default=3.0,
                        help='环绕轨迹半径')
    parser.add_argument('--num_views', type=int, default=120,
                        help='环绕轨迹视角数量')
    parser.add_argument('--height', type=float, default=1.2,
                        help='相机高度')
    parser.add_argument('--n_samples', type=int, default=1024,
                        help='每条光线的采样点数')
    
    # 解析命令行参数
    cmd_args = parser.parse_args()
    
    # 解析配置文件
    args = config_parser(['--config', cmd_args.config])
    
    # 设置检查点路径
    if cmd_args.ckpt:
        ckpt_path = cmd_args.ckpt
    else:
        # 自动查找检查点文件
        log_dir = f"log/{args.expname}"
        ckpt_path = f"{log_dir}/{args.expname}.th"
        if not os.path.exists(ckpt_path):
            # 尝试其他可能的路径
            possible_paths = [
                f"{log_dir}/tensorf_lego_VM.th",
                f"{log_dir}/tensorf_lego_VM/tensorf_lego_VM.th"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    ckpt_path = path
                    break
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("NeRF 测试和渲染脚本")
    print("=" * 60)
    print(f"配置文件: {cmd_args.config}")
    print(f"检查点文件: {ckpt_path}")
    print(f"设备: {device}")
    print(f"数据集: {args.dataset_name}")
    print(f"数据路径: {args.datadir}")
    
    # 检查文件是否存在
    if not os.path.exists(ckpt_path):
        print(f"错误：找不到检查点文件 {ckpt_path}")
        print("请确保训练已完成，或手动指定检查点文件路径")
        return
    
    # 加载数据集
    print("\n1. 加载数据集...")
    try:
        dataset = dataset_dict[args.dataset_name]
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True)
        white_bg = test_dataset.white_bg
        ndc_ray = args.ndc_ray
        print(f"   数据集加载成功，测试图像数量: {len(test_dataset.all_rays)}")
    except Exception as e:
        print(f"错误：加载数据集失败 - {e}")
        return
    
    # 加载训练好的模型
    print("\n2. 加载训练好的模型...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        
        model_name = args.model_name
        tensorf = eval(model_name)(**kwargs)
        tensorf.load(ckpt)
        tensorf.eval()
        print(f"   模型加载成功: {model_name}")
    except Exception as e:
        print(f"错误：加载模型失败 - {e}")
        return
    
    # 创建输出目录
    if cmd_args.output_dir:
        logfolder = cmd_args.output_dir
    else:
        logfolder = os.path.dirname(ckpt_path)
    
    os.makedirs(f'{logfolder}/test_results', exist_ok=True)
    os.makedirs(f'{logfolder}/render_path', exist_ok=True)
    print(f"   输出目录: {logfolder}")
    
    # 3. 在测试集上进行定量评价
    print("\n3. 在测试集上进行定量评价...")
    print("   计算PSNR、SSIM、LPIPS等指标...")
    
    try:
        PSNRs_test = evaluation(
            test_dataset, 
            tensorf, 
            args, 
            OctreeRender_trilinear_fast, 
            f'{logfolder}/test_results',
            N_vis=-1,  # 渲染所有测试图像
            N_samples=cmd_args.n_samples,
            white_bg=white_bg, 
            ndc_ray=ndc_ray, 
            device=device,
            compute_extra_metrics=True  # 计算额外指标
        )
        
        # 读取并显示结果
        mean_file = f'{logfolder}/test_results/mean.txt'
        if os.path.exists(mean_file):
            metrics = np.loadtxt(mean_file)
            if len(metrics) == 4:
                psnr, ssim, lpips_alex, lpips_vgg = metrics
                print(f"\n测试结果:")
                print(f"  PSNR: {psnr:.2f} dB")
                print(f"  SSIM: {ssim:.4f}")
                print(f"  LPIPS (Alex): {lpips_alex:.4f}")
                print(f"  LPIPS (VGG): {lpips_vgg:.4f}")
            else:
                psnr = metrics[0]
                print(f"\n测试结果:")
                print(f"  PSNR: {psnr:.2f} dB")
        
        print(f"   测试图像已保存到: {logfolder}/test_results/")
        
    except Exception as e:
        print(f"错误：测试评价失败 - {e}")
        return
    
    # 4. 渲染环绕视频
    print("\n4. 渲染环绕视频...")
    print(f"   生成环绕轨迹 (半径={cmd_args.radius}, 视角数={cmd_args.num_views})...")
    
    try:
        # 创建环绕轨迹
        c2ws = create_circular_path(
            radius=cmd_args.radius, 
            num_views=cmd_args.num_views, 
            height=cmd_args.height
        )
        
        # 渲染环绕视频
        evaluation_path(
            test_dataset=test_dataset,
            tensorf=tensorf,
            c2ws=c2ws,
            renderer=OctreeRender_trilinear_fast,
            savePath=f'{logfolder}/render_path',
            N_vis=-1,
            prtx='',
            N_samples=cmd_args.n_samples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            compute_extra_metrics=False,  # 环绕视频不需要计算指标
            device=device
        )
        
        print(f"   环绕视频已保存到: {logfolder}/render_path/")
        
    except Exception as e:
        print(f"错误：渲染环绕视频失败 - {e}")
        return
    
    # 5. 生成测试报告
    print("\n5. 生成测试报告...")
    report_file = f'{logfolder}/test_report.txt'
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("NeRF 测试报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"数据集: {args.dataset_name}\n")
            f.write(f"数据路径: {args.datadir}\n")
            f.write(f"模型: {args.model_name}\n")
            f.write(f"检查点: {ckpt_path}\n")
            f.write(f"设备: {device}\n\n")
            
            f.write("渲染参数:\n")
            f.write(f"  采样点数: {cmd_args.n_samples}\n")
            f.write(f"  环绕半径: {cmd_args.radius}\n")
            f.write(f"  视角数量: {cmd_args.num_views}\n")
            f.write(f"  相机高度: {cmd_args.height}\n\n")
            
            if os.path.exists(mean_file):
                metrics = np.loadtxt(mean_file)
                if len(metrics) == 4:
                    psnr, ssim, lpips_alex, lpips_vgg = metrics
                    f.write("定量评价结果:\n")
                    f.write(f"  PSNR: {psnr:.2f} dB\n")
                    f.write(f"  SSIM: {ssim:.4f}\n")
                    f.write(f"  LPIPS (Alex): {lpips_alex:.4f}\n")
                    f.write(f"  LPIPS (VGG): {lpips_vgg:.4f}\n")
                else:
                    psnr = metrics[0]
                    f.write("定量评价结果:\n")
                    f.write(f"  PSNR: {psnr:.2f} dB\n")
            
            f.write(f"\n输出文件:\n")
            f.write(f"  测试图像: {logfolder}/test_results/\n")
            f.write(f"  环绕视频: {logfolder}/render_path/\n")
            f.write(f"  测试报告: {report_file}\n")
        
        print(f"   测试报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"错误：生成测试报告失败 - {e}")
    
    print("\n" + "=" * 60)
    print("测试和渲染完成！")
    print("=" * 60)
    print(f"测试结果保存在: {logfolder}/test_results/")
    print(f"环绕视频保存在: {logfolder}/render_path/")
    print(f"  - video.mp4: RGB视频")
    print(f"  - depthvideo.mp4: 深度视频")
    print(f"测试报告: {report_file}")

if __name__ == '__main__':
    main() 