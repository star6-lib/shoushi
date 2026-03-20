"""
训练日志可视化脚本
解析 output-*.log 文件，绘制损失曲线并分析训练状态。

使用方法：
    python plot_training_log.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_latest_log(log_dir: Path) -> Path | None:
    """在指定目录中查找最新的 output-*.log 文件"""
    logs = sorted(log_dir.glob("output-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_log(log_path: Path) -> dict:
    """解析训练日志，提取各项指标"""
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    # 提取配置信息
    config = {}
    for line in lines:
        if line.startswith("batchsize:"):
            config["batchsize"] = int(line.split(":")[1].strip())
        elif line.startswith("learning rate:"):
            config["learning_rate"] = float(line.split(":")[1].strip())
        elif line.startswith("Using dataset:"):
            config["dataset"] = line.split(":")[1].strip()
        elif line.startswith("Using model:"):
            config["model"] = line.split(":")[1].strip()
        elif "Best L2 relative error" in line:
            config["best_test_error"] = float(line.split()[-1])

    # 提取每个 epoch 的损失
    epochs = []
    for i, line in enumerate(lines):
        if line.startswith("Current epoch error:"):
            try:
                epoch_error = float(line.split(":")[1].strip())
                # 后续几行应该是各种损失
                pde_loss = fix_bc = free_bc = load_bc = None

                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j]
                    if "pde loss:" in next_line.lower():
                        pde_loss = float(next_line.split(":")[1].strip())
                    elif "fix bc loss:" in next_line.lower():
                        fix_bc = float(next_line.split(":")[1].strip())
                    elif "free bc loss:" in next_line.lower():
                        free_bc = float(next_line.split(":")[1].strip())
                    elif "load bc loss:" in next_line.lower():
                        load_bc = float(next_line.split(":")[1].strip())

                # 跳过 inf 值（初始化阶段）
                if pde_loss is not None and pde_loss < 1e10:
                    epochs.append({
                        "epoch": len(epochs) + 1,
                        "error": epoch_error,
                        "pde_loss": pde_loss,
                        "fix_bc_loss": fix_bc,
                        "free_bc_loss": free_bc,
                        "load_bc_loss": load_bc,
                    })
            except (ValueError, IndexError):
                continue

    return {
        "config": config,
        "epochs": epochs,
    }


def analyze_training(data: dict) -> dict:
    """分析训练状态，判断是否收敛"""
    epochs = data["epochs"]
    if not epochs:
        return {"error": "No valid epoch data found"}

    errors = [e["error"] for e in epochs]
    pde_losses = [e["pde_loss"] for e in epochs if e["pde_loss"] is not None]
    fix_bc_losses = [e["fix_bc_loss"] for e in epochs if e["fix_bc_loss"] is not None]

    # 基本统计
    n_epochs = len(epochs)
    initial_error = errors[0]
    final_error = errors[-1]
    best_error = min(errors)

    # 计算收敛指标
    # 1. 损失下降比例
    error_reduction = (initial_error - final_error) / initial_error * 100

    # 2. 最后 10% 轮次的波动（标准差/均值）
    last_10pct = max(1, int(n_epochs * 0.1))
    last_errors = errors[-last_10pct:]
    fluctuation = np.std(last_errors) / (np.mean(last_errors) + 1e-8) * 100

    # 3. 最近 20 个 epoch 的趋势（线性拟合斜率）
    recent = errors[-20:]
    if len(recent) >= 2:
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        trend = "decreasing" if slope < -0.001 else "increasing" if slope > 0.001 else "stable"
    else:
        slope = 0
        trend = "stable"

    # 4. 判断是否收敛
    is_converged = fluctuation < 5 and trend != "increasing"

    return {
        "n_epochs": n_epochs,
        "initial_error": initial_error,
        "final_error": final_error,
        "best_error": best_error,
        "error_reduction_pct": error_reduction,
        "final_fluctuation_pct": fluctuation,
        "recent_trend": trend,
        "recent_slope": slope,
        "is_converged": is_converged,
    }


def plot_losses(data: dict, output_dir: Path):
    """绘制损失曲线"""
    epochs = data["epochs"]
    if not epochs:
        print("No epoch data to plot")
        return

    epoch_nums = [e["epoch"] for e in epochs]
    errors = [e["error"] for e in epochs]
    pde_losses = [e["pde_loss"] for e in epochs]
    fix_bc_losses = [e["fix_bc_loss"] for e in epochs]
    free_bc_losses = [e["free_bc_loss"] for e in epochs]
    load_bc_losses = [e["load_bc_loss"] for e in epochs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 总误差曲线
    ax1 = axes[0, 0]
    ax1.plot(epoch_nums, errors, 'b-', linewidth=1.5, label='Epoch Error')
    ax1.axhline(y=min(errors), color='r', linestyle='--', alpha=0.7, label=f'Best: {min(errors):.4f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Relative L2 Error')
    ax1.set_title('Training Error (L2 Relative Error)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. PDE Loss
    ax2 = axes[0, 1]
    ax2.plot(epoch_nums, pde_losses, 'g-', linewidth=1.5, label='PDE Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PDE Loss')
    ax2.set_title('PDE Residual Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 边界条件损失（BC Losses）
    ax3 = axes[1, 0]
    ax3.plot(epoch_nums, fix_bc_losses, 'r-', linewidth=1.5, label='Fix BC Loss')
    ax3.plot(epoch_nums, free_bc_losses, 'orange', linewidth=1.5, label='Free BC Loss')
    ax3.plot(epoch_nums, load_bc_losses, 'purple', linewidth=1.5, label='Load BC Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('BC Loss')
    ax3.set_title('Boundary Condition Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 所有损失对数图
    ax4 = axes[1, 1]
    ax4.plot(epoch_nums, pde_losses, 'g-', linewidth=1.5, label='PDE Loss', alpha=0.8)
    ax4.plot(epoch_nums, fix_bc_losses, 'r-', linewidth=1.5, label='Fix BC Loss', alpha=0.8)
    ax4.plot(epoch_nums, free_bc_losses, 'orange', linewidth=1.5, label='Free BC Loss', alpha=0.8)
    ax4.plot(epoch_nums, load_bc_losses, 'purple', linewidth=1.5, label='Load BC Loss', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('All Losses (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()

    # 保存图片
    fig_path = output_dir / "training_loss_plot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {fig_path}")

    # 同时保存 PDF 矢量图
    pdf_path = output_dir / "training_loss_plot.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF 图表已保存到: {pdf_path}")

    plt.close()


def main():
    # 查找日志文件
    script_dir = Path(__file__).resolve().parent
    log_file = find_latest_log(script_dir)

    if log_file is None:
        # 尝试在上级目录查找
        log_file = find_latest_log(script_dir.parent)
        if log_file is None:
            print("未找到 output-*.log 文件")
            sys.exit(1)

    print(f"使用日志文件: {log_file}")

    # 解析日志
    data = parse_log(log_file)

    # 保存解析后的数据
    json_path = log_file.with_suffix(".json")
    json_path = json_path.with_name(json_path.name.replace("output-", "parsed_"))
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"解析数据已保存到: {json_path}")

    # Analyze training status
    analysis = analyze_training(data)
    print("\n" + "=" * 60)
    print("Training Status Analysis")
    print("=" * 60)
    print(f"Total epochs: {analysis['n_epochs']}")
    print(f"Initial error: {analysis['initial_error']:.6f}")
    print(f"Final error: {analysis['final_error']:.6f}")
    print(f"Best error: {analysis['best_error']:.6f}")
    print(f"Error reduction: {analysis['error_reduction_pct']:.2f}%")
    print(f"Last 10% fluctuation: {analysis['final_fluctuation_pct']:.2f}%")
    print(f"Recent 20 epochs trend: {analysis['recent_trend']} (slope={analysis['recent_slope']:.6f})")
    print(f"Converged: {'Yes' if analysis['is_converged'] else 'No'}")
    print("=" * 60)

    # 绘制图表
    plot_losses(data, script_dir)

    print("\n=== Training Conclusion ===")
    if analysis["is_converged"]:
        print("Model has converged, training completed.")
        print(f"Best test error: {analysis['best_error']:.4f} ({analysis['best_error']*100:.2f}%)")
    else:
        print("Model has NOT fully converged yet. Consider:")
        print("  - Continue training for more epochs")
        print("  - Adjust learning rate")
        print("  - Check data quality")


if __name__ == "__main__":
    main()
