import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.ticker as ticker
import os

# 1. 解析日志文件
# 解决报错的关键：加入 encoding='utf-8'
log_file = 'output-58547971.log'

epochs = []
val_l2_u, val_l2_v, val_l2_vm = [], [], []
total_loss = []
loss_u, loss_v, loss_sxx, loss_syy, loss_sxy = [], [], [], [], []
w_u, w_v, w_sxx, w_syy, w_sxy = [], [], [], [], []

print(f"Reading {log_file} with utf-8 encoding...")

with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # 解析 Epoch 和 验证集误差
    if line.startswith('Epoch ') and 'Validation L2 Error' in line:
        match = re.search(r'Epoch (\d+) - Validation L2 Error \| U: ([\d.]+) \| V: ([\d.]+) \| VM: ([\d.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            val_l2_u.append(float(match.group(2)))
            val_l2_v.append(float(match.group(3)))
            val_l2_vm.append(float(match.group(4)))

    # 解析总 Loss
    elif 'Total MSE Loss:' in line:
        match = re.search(r'Total MSE Loss:\s+([\d.]+)', line)
        if match: total_loss.append(float(match.group(1)))

    # 解析各项分 Loss 和 NTK 权重
    elif '├─ Loss U:' in line:
        match = re.search(r'Loss U:\s+([\d.]+)\s+\|\s+Weight\(λ_u\):\s+([\d.]+)', line)
        if match:
            loss_u.append(float(match.group(1)))
            w_u.append(float(match.group(2)))

    elif '├─ Loss V:' in line:
        match = re.search(r'Loss V:\s+([\d.]+)\s+\|\s+Weight\(λ_v\):\s+([\d.]+)', line)
        if match:
            loss_v.append(float(match.group(1)))
            w_v.append(float(match.group(2)))

    elif '├─ Loss Sxx:' in line:
        match = re.search(r'Loss Sxx:\s+([\d.]+)\s+\|\s+Weight\(λ_sxx\):\s+([\d.]+)', line)
        if match:
            loss_sxx.append(float(match.group(1)))
            w_sxx.append(float(match.group(2)))

    elif '├─ Loss Syy:' in line:
        match = re.search(r'Loss Syy:\s+([\d.]+)\s+\|\s+Weight\(λ_syy\):\s+([\d.]+)', line)
        if match:
            loss_syy.append(float(match.group(1)))
            w_syy.append(float(match.group(2)))

    elif '└─ Loss Sxy:' in line:
        match = re.search(r'Loss Sxy:\s+([\d.]+)\s+\|\s+Weight\(λ_sxy\):\s+([\d.]+)', line)
        if match:
            loss_sxy.append(float(match.group(1)))
            w_sxy.append(float(match.group(2)))

print(f"Successfully parsed {len(epochs)} epochs.")

# 防止有些列表长度不一致报错，截断到最短长度
min_len = min(len(epochs), len(total_loss), len(loss_u))
epochs = epochs[:min_len]
total_loss = total_loss[:min_len]
val_l2_vm = val_l2_vm[:min_len]
loss_u, loss_v = loss_u[:min_len], loss_v[:min_len]
loss_sxx, loss_syy, loss_sxy = loss_sxx[:min_len], loss_syy[:min_len], loss_sxy[:min_len]
w_u, w_sxx = w_u[:min_len], w_sxx[:min_len]
w_syy, w_sxy = w_syy[:min_len], w_sxy[:min_len]

# ================= 论文级可视化配置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'stix'

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# --- 子图 1: Total Loss 与 Validation VM Error ---
ax1.plot(epochs, total_loss, 'b-', linewidth=2, label='Total MSE Loss')
ax1.set_ylabel('Total MSE Loss', color='b', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('log')
ax1.grid(True, linestyle='--', alpha=0.6)

ax1_twin = ax1.twinx()
ax1_twin.plot(epochs, val_l2_vm, 'r--', linewidth=2, label='Val Error (Von Mises)')
ax1_twin.set_ylabel('Relative $L_2$ Error', color='r', fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor='r')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax1.set_title('(a) Overall Training and Validation Performance', loc='left', fontsize=12, fontweight='bold')

# --- 子图 2: 分项 Loss (Log 坐标) ---
ax2.plot(epochs, loss_u, linestyle='-', color='#1f77b4', label=r'$\mathcal{L}_U$')
ax2.plot(epochs, loss_v, linestyle='-', color='#ff7f0e', label=r'$\mathcal{L}_V$')
ax2.plot(epochs, loss_sxx, linestyle='-', color='#2ca02c', label=r'$\mathcal{L}_{\sigma_{xx}}$')
ax2.plot(epochs, loss_syy, linestyle='-', color='#d62728', label=r'$\mathcal{L}_{\sigma_{yy}}$')
ax2.plot(epochs, loss_sxy, linestyle='-', color='#9467bd', label=r'$\mathcal{L}_{\sigma_{xy}}$')
ax2.set_ylabel('Component Loss (Log)', fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.0))
ax2.set_title('(b) Component MSE Losses', loc='left', fontsize=12, fontweight='bold')

# --- 子图 3: NTK 权重变化 ---
ax3.plot(epochs, w_u, linestyle='-', color='#1f77b4', label=r'$\lambda_U$')
ax3.plot(epochs, w_sxx, linestyle='-', color='#2ca02c', label=r'$\lambda_{\sigma_{xx}}$')
ax3.plot(epochs, w_syy, linestyle='-', color='#d62728', label=r'$\lambda_{\sigma_{yy}}$')
ax3.plot(epochs, w_sxy, linestyle='-', color='#9467bd', label=r'$\lambda_{\sigma_{xy}}$')
ax3.set_xlabel('Epochs', fontweight='bold', fontsize=12)
ax3.set_ylabel('NTK Weights $\lambda$', fontweight='bold')
ax3.set_yscale('log')  # 权重差异太大，使用对数坐标
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(ncol=4, loc='upper center')
ax3.set_title('(c) Evolution of Adaptive NTK Weights', loc='left', fontsize=12, fontweight='bold')

plt.tight_layout()

save_path = 'training_analysis.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {save_path}")
plt.show()