import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(file_path):
    """从 log 文件中提取 loss 和 error 数据"""
    data = {
        'epoch_idx': [],
        'error': [],
        'pde': [],
        'ce': [],
        'fix': [],
        'free': [],
        'load': []
    }

    # 正则表达式匹配模式
    regex_err = re.compile(r'Current epoch error:\s+([0-9\.\-eE]+|inf)')
    regex_pde = re.compile(r'current epochs pde loss:\s+([0-9\.\-eE]+|inf)')
    regex_ce = re.compile(r'CE \(Hooke\):\s+([0-9\.\-eE]+|inf)')
    regex_fix = re.compile(r'fix bc loss:\s+([0-9\.\-eE]+|inf)')
    regex_free = re.compile(r'free bc loss:\s+([0-9\.\-eE]+|inf)')
    regex_load = re.compile(r'load bc loss:\s+([0-9\.\-eE]+|inf)')

    step = 1

    with open(file_path, 'r', encoding='utf-8') as f:
        # 为了应对分行打印，我们按行读取并暂存
        current_step_data = {}
        for line in f:
            if match := regex_err.search(line):
                current_step_data['error'] = float(match.group(1)) if match.group(1) != 'inf' else np.nan
            elif match := regex_pde.search(line):
                current_step_data['pde'] = float(match.group(1)) if match.group(1) != 'inf' else np.nan
            elif match := regex_ce.search(line):
                current_step_data['ce'] = float(match.group(1)) if match.group(1) != 'inf' else np.nan
            elif match := regex_fix.search(line):
                current_step_data['fix'] = float(match.group(1)) if match.group(1) != 'inf' else np.nan
            elif match := regex_free.search(line):
                current_step_data['free'] = float(match.group(1)) if match.group(1) != 'inf' else np.nan
            elif match := regex_load.search(line):
                current_step_data['load'] = float(match.group(1)) if match.group(1) != 'inf' else np.nan

                # 当 load 读取到时，说明这一轮的数据收集完毕 (根据您的日志格式)
                if all(k in current_step_data for k in ['error', 'pde', 'ce', 'fix', 'free', 'load']):
                    data['epoch_idx'].append(step)
                    for k in ['error', 'pde', 'ce', 'fix', 'free', 'load']:
                        data[k].append(current_step_data[k])
                    current_step_data = {}
                    step += 1

    return data


def plot_academic_curves(data, save_prefix='training'):
    """绘制学术级的高质量图表"""

    # 学术论文全局字体与样式设置
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'legend.fontsize': 12,
        'legend.framealpha': 0.9,
    })

    epochs = np.array(data['epoch_idx']) * 2  # 假设 visual_freq = 2

    # ==========================================
    # 图 1: 物理损失函数 (Log Scale)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(epochs, data['load'], label=r'$\mathcal{L}_{load}$ (Force BC)', color='#d62728', linewidth=2.5, alpha=0.85)
    ax1.plot(epochs, data['fix'], label=r'$\mathcal{L}_{fix}$ (Fixed BC)', color='#ff7f0e', linewidth=2.5, alpha=0.85)
    ax1.plot(epochs, data['free'], label=r'$\mathcal{L}_{free}$ (Hole & Free BC)', color='#2ca02c', linewidth=2.5,
             alpha=0.85)
    ax1.plot(epochs, data['pde'], label=r'$\mathcal{L}_{pde}$ (Equilibrium)', color='#1f77b4', linewidth=2.5,
             alpha=0.85)
    ax1.plot(epochs, data['ce'], label=r'$\mathcal{L}_{ce}$ (Constitutive)', color='#9467bd', linewidth=2.5, alpha=0.85)

    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Loss Value (Log Scale)', fontweight='bold')
    ax1.set_title('Convergence of Physical Constraints', fontweight='bold', pad=15)

    # 开启网格线 (主刻度和次刻度)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='upper right', edgecolor='black')

    plt.tight_layout()
    fig1.savefig(f'{save_prefix}_loss_components.png', dpi=300, bbox_inches='tight')
    fig1.savefig(f'{save_prefix}_loss_components.pdf', format='pdf', bbox_inches='tight')  # 投递顶会常用的矢量图格式
    plt.close(fig1)

    # ==========================================
    # 图 2: 相对误差 (Relative L2 Error)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.plot(epochs, data['error'], label='Relative $L_2$ Error', color='#17becf', linewidth=3.0)

    ax2.set_xlabel('Epochs', fontweight='bold')
    ax2.set_ylabel('Relative Error', fontweight='bold')
    ax2.set_title('Model Prediction Error over Training', fontweight='bold', pad=15)

    ax2.grid(True, ls="--", alpha=0.7)
    ax2.legend(loc='upper right', edgecolor='black')

    plt.tight_layout()
    fig2.savefig(f'{save_prefix}_l2_error.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{save_prefix}_l2_error.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig2)

    print(f"✅ 图表已成功生成！已保存为 {save_prefix}_loss_components.png/pdf 和 {save_prefix}_l2_error.png/pdf")


if __name__ == "__main__":
    # 请将这里的 'output.log' 替换为您真实的日志文件名
    LOG_FILE_PATH = 'output-8456737.log'

    try:
        parsed_data = parse_log_file(LOG_FILE_PATH)
        if len(parsed_data['epoch_idx']) == 0:
            print("警告：未能从日志中提取到完整数据，请检查日志格式是否匹配。")
        else:
            print(f"成功提取 {len(parsed_data['epoch_idx'])} 个评估步的数据。开始绘图...")
            plot_academic_curves(parsed_data, save_prefix='LA_PIDON')
    except Exception as e:
        print(f"发生错误: {e}")