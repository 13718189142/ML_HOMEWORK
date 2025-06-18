# ML_HOMEWORK
# TensoRF: 基于张量分解的神经辐射场

本项目实现了TensoRF（Tensor Radiance Fields）算法，用于3D场景重建和新视角合成。通过张量分解技术，TensoRF能够高效地表示3D场景，实现高质量的神经辐射场渲染。

## 📋 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [参考文献](#参考文献)

## 🎯 项目简介

TensoRF是一种基于张量分解的神经辐射场方法，相比传统的NeRF具有以下优势：

- **高效表示**：使用张量分解技术，大幅减少内存占用
- **快速训练**：训练速度比传统NeRF快10-100倍
- **高质量渲染**：保持与NeRF相当的渲染质量
- **灵活配置**：支持多种数据集和场景类型

## 🔧 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (推荐)
- 其他依赖包见 `requirements.txt`

### 推荐配置
- GPU: NVIDIA RTX 3080 或更高
- 内存: 16GB+
- 存储: 50GB+ 可用空间

## 📦 安装指南

1. **克隆项目**
```bash
git clone <repository-url>
cd TensoRF-main
```

2. **创建虚拟环境**
```bash
conda create -n tensorf python=3.8
conda activate tensorf
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **验证安装**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 数据准备

### 支持的数据集

1. **NeRF Synthetic** (推荐用于测试)
   - 包含: lego, ship, mic, chair, drums, ficus, hotdog, materials
   - 下载: [NeRF Synthetic Dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZNJQSlX9a1)

2. **LLFF (Local Light Field Fusion)**
   - 包含: fern, flower, room, leaves, horns, trex, fortress, orchids
   - 下载: [LLFF Dataset](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7)

3. **自定义数据**
   - 支持COLMAP格式的相机参数
   - 参考 `dataLoader/your_own_data.py`

### 数据组织结构
```
data/
├── nerf_synthetic/
│   └── lego/
│       ├── train/
│       ├── test/
│       ├── val/
│       ├── transforms_train.json
│       ├── transforms_test.json
│       └── transforms_val.json
└── nerf_llff_data/
    └── fern/
        ├── images/
        ├── poses_bounds.npy
        └── ...
```

## 🚀 使用方法

### 1. 训练模型

#### 基本训练命令
```bash
# 使用配置文件训练
python train.py --config configs/lego.txt

# 自定义参数训练
python train.py \
    --config configs/lego.txt \
    --n_iters 10000 \
    --batch_size 4096 \
    --lr_init 0.02
```

#### 训练参数说明
- `--n_iters`: 训练迭代次数 (默认: 10000)
- `--batch_size`: 批处理大小 (默认: 4096)
- `--lr_init`: 初始学习率 (默认: 0.02)
- `--progress_refresh_rate`: 进度显示频率 (默认: 100)

#### 训练过程监控
- 训练日志保存在 `log/{expname}/`
- 训练曲线图自动生成
- TensorBoard支持: `tensorboard --logdir=log/`

### 2. 测试和渲染

#### 运行测试脚本
```bash
# 基本测试
python test_and_render.py --config configs/lego.txt

# 指定模型文件
python test_and_render.py \
    --config configs/lego.txt \
    --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th

# 自定义渲染参数
python test_and_render.py \
    --config configs/lego.txt \
    --radius 4.0 \
    --num_views 180 \
    --n_samples 2048
```

#### 测试参数说明
- `--ckpt`: 模型检查点文件路径
- `--radius`: 环绕轨迹半径 (默认: 3.0)
- `--num_views`: 环绕视角数量 (默认: 120)
- `--height`: 相机高度 (默认: 1.2)
- `--n_samples`: 采样点数 (默认: 1024)

### 3. 可视化训练曲线

```bash
# 生成训练曲线图
python plot_training_curves.py
```

## 📊 实验结果

### 测试结果示例 (Lego数据集)

| 指标 | 数值 | 评价 |
|------|------|------|
| PSNR | 34.16 dB | 优秀 |
| SSIM | 0.9748 | 极优秀 |
| LPIPS (Alex) | 0.0138 | 极优秀 |
| LPIPS (VGG) | 0.0292 | 极优秀 |

### 性能对比

| 方法 | PSNR | 训练时间 | 内存占用 |
|------|------|----------|----------|
| NeRF | ~33 dB | 12-24小时 | 8GB+ |
| **TensoRF** | **~34 dB** | **1-2小时** | **2-4GB** |

## 📁 项目结构

```
TensoRF-main/
├── configs/                 # 配置文件
│   ├── lego.txt            # Lego数据集配置
│   ├── flower.txt          # Flower数据集配置
│   └── ...
├── dataLoader/             # 数据加载器
│   ├── blender.py          # NeRF Synthetic数据集
│   ├── llff.py             # LLFF数据集
│   └── your_own_data.py    # 自定义数据
├── models/                 # 模型定义
│   ├── tensorBase.py       # 基础张量模型
│   └── tensoRF.py          # TensoRF实现
├── log/                    # 训练日志和结果
│   └── tensorf_lego_VM/    # 实验输出
├── train.py                # 训练脚本
├── test_and_render.py      # 测试和渲染脚本
├── plot_training_curves.py # 训练曲线绘制
└── README.md               # 项目说明
```

## ⚙️ 配置说明

### 主要配置参数

```bash
# 数据集配置
dataset_name = blender          # 数据集类型
datadir = ./data/nerf_synthetic/lego  # 数据路径

# 训练配置
n_iters = 10000                 # 训练迭代次数
batch_size = 4096               # 批处理大小
lr_init = 0.02                  # 初始学习率

# 模型配置
model_name = TensorVMSplit      # 模型类型
N_voxel_init = 2097156          # 初始体素数 (128^3)
N_voxel_final = 27000000        # 最终体素数 (300^3)

# 渲染配置
N_vis = 5                       # 可视化图像数量
vis_every = 5000                # 可视化频率
```

## 📈 训练监控

### 实时监控
```bash
# 查看训练进度
tail -f log/tensorf_lego_VM/training_metrics.txt

# 启动TensorBoard
tensorboard --logdir=log/tensorf_lego_VM/
```

### 关键指标
- **PSNR**: 图像质量指标，越高越好
- **Loss**: 训练损失，应该逐渐下降
- **Memory**: GPU内存使用情况

## 🎬 输出结果

### 训练完成后会生成：
1. **模型文件**: `tensorf_lego_VM.th`
2. **训练曲线**: `training_curves_combined.png`
3. **测试图像**: `test_results/`
4. **环绕视频**: `render_path/video.mp4`
5. **测试报告**: `test_report.txt`

## 📚 参考文献

1. Chen, A., Xu, Z., Geiger, A., Yu, J., & Su, H. (2022). TensoRF: Tensorial Radiance Fields. In European Conference on Computer Vision (ECCV).

2. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV).
# NeRF-PyTorch 中文说明

## nerf-pytorch-master
NeRF (Neural Radiance Fields) 是一种用于新视角合成的先进方法。本仓库是 NeRF 的 PyTorch 实现，复现了原论文结果且运行速度更快。

## 安装
```bash
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

## 开始

1. 训练模型：
```bash
python run_nerf.py --config configs/lego.txt
```

2. 渲染测试：
```bash
python run_nerf.py --config configs/lego.txt --render_only
```


