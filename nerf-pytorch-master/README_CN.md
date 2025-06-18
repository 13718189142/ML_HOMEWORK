# NeRF-PyTorch 中文说明

## 项目简介
NeRF (Neural Radiance Fields) 是一种用于新视角合成的先进方法。本仓库是 NeRF 的 PyTorch 实现，复现了原论文结果且运行速度更快。

## 安装
```bash
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

## 快速开始
1. 下载示例数据：
```bash
bash download_example_data.sh
```

2. 训练模型：
```bash
python run_nerf.py --config configs/lego.txt
```

3. 渲染测试：
```bash
python run_nerf.py --config configs/lego.txt --render_only
```

## 数据集
- 示例数据：`lego`, `fern` 等
- 更多数据集：[下载链接](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- 预训练模型：[下载链接](https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv)

## 配置说明
- 配置文件位于 `configs/` 目录
- 支持的数据集：`lego`, `ship`, `fern`, `flower`, `horns`, `trex` 等
- 训练时间：约4-8小时（单张2080Ti）

## 引用
```bibtex
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
``` 