import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取训练指标数据
data = pd.read_csv('log/tensorf_lego_VM/training_metrics.txt')

# 提取数据
iterations = data.iloc[:, 0]  # 第一列：迭代数
psnr = data.iloc[:, 1]        # 第二列：PSNR
loss = data.iloc[:, 2]        # 第三列：Loss

# 创建图形和双Y轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制PSNR曲线（红色，左Y轴）
color1 = 'red'
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('PSNR (dB)', color=color1, fontsize=12)
line1 = ax1.plot(iterations, psnr, color=color1, linewidth=2, label='PSNR')
ax1.tick_params(axis='y', labelcolor=color1)

# 创建右Y轴，绘制Loss曲线（蓝色）
ax2 = ax1.twinx()
color2 = 'blue'
ax2.set_ylabel('Loss', color=color2, fontsize=12)
line2 = ax2.plot(iterations, loss, color=color2, linewidth=2, label='Loss')
ax2.tick_params(axis='y', labelcolor=color2)

# 添加网格
ax1.grid(True, alpha=0.3)

# 设置标题
plt.title('Training PSNR and Loss over Iterations', fontsize=14, pad=20)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('log/tensorf_lego_VM/training_curves_combined.png', dpi=300, bbox_inches='tight')
plt.show()

print("训练曲线图已保存到: log/tensorf_lego_VM/training_curves_combined.png") 