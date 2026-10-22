"""Generate comprehensive summary figures for Lab 4 report."""
from __future__ import annotations

import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copy best model figures to outputs/figures/
best_run_figures = OUTPUTS_DIR / "runs" / "stacked_bigru" / "figures"
if best_run_figures.exists():
    for fig_name in ["confusion_matrix.png", "loss_curve.png", "metric_curve.png"]:
        src = best_run_figures / fig_name
        if src.exists():
            shutil.copy2(src, FIGURES_DIR / fig_name)
            print(f"Copied {src} -> {FIGURES_DIR / fig_name}")

# 2. Generate Ablation Comparison Chart
models = [
    "Lab 3 Vanilla RNN\n(Baseline)",
    "Lab 3 Vanilla RNN\n(Best compact)",
    "Baseline LSTM\n(1L, 1-dir)",
    "GRU Baseline\n(1L, 1-dir)",
    "BiLSTM\n(1L, 2-dir)",
    "Stacked BiGRU\n(2L, 2-dir)",
]
test_acc = [65.28, 71.51, 79.59, 85.34, 82.55, 85.34]
val_f1 = [64.55, 71.45, 79.77, 86.37, 84.00, 86.88]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
rects1 = ax.bar(x - width/2, val_f1, width, label="Val Macro-F1 (%)", color="#3b82f6", alpha=0.9)
rects2 = ax.bar(x + width/2, test_acc, width, label="Test Accuracy (%)", color="#10b981", alpha=0.9)

ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
ax.set_title("CSC4007 Lab 4: Model Ablation & Progress over Lab 3", fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.5)
ax.set_ylim(50, 95)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
ablation_path = FIGURES_DIR / "ablation_comparison.png"
plt.savefig(ablation_path, dpi=160)
plt.close()
print(f"Generated {ablation_path}")

# 3. Generate Learning Curves Comparison (Val Loss & Val F1 across epochs)
runs = {
    "baseline_lstm": ("Baseline LSTM", "#f97316"),
    "gru_baseline": ("GRU Baseline", "#10b981"),
    "bilstm": ("BiLSTM", "#8b5cf6"),
    "stacked_bigru": ("Stacked BiGRU (Best)", "#ef4444"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=160)

for run_key, (label, color) in runs.items():
    csv_file = OUTPUTS_DIR / "runs" / run_key / "metrics" / "epoch_history.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        ax1.plot(df["epoch"], df["val_loss"], marker="o", linewidth=2, label=label, color=color)
        ax2.plot(df["epoch"], df["val_macro_f1"] * 100, marker="s", linewidth=2, label=label, color=color)

ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
ax1.set_ylabel("Validation Loss", fontsize=10, fontweight="bold")
ax1.set_title("Validation Loss across Epochs", fontsize=11, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
ax2.set_ylabel("Validation Macro-F1 (%)", fontsize=10, fontweight="bold")
ax2.set_title("Validation Macro-F1 (%) across Epochs", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.suptitle("CSC4007 Lab 4: Learning Curves Comparison across All 4 Architectures", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
learning_path = FIGURES_DIR / "learning_curves_comparison.png"
plt.savefig(learning_path, dpi=160)
plt.close()
print(f"Generated {learning_path}")

# 4. Generate Side-by-Side Confusion Matrix (Baseline LSTM vs Stacked BiGRU)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)

# Baseline LSTM confusion matrix
# From metrics summary:
# Baseline LSTM: TN: 9949, FP: 2551, FN: 2552, TP: 9948
cm_baseline = np.array([[9949, 2551], [2552, 9948]])
im1 = ax1.imshow(cm_baseline, cmap="Blues", interpolation="nearest")
ax1.set_title("Baseline LSTM (Acc: 79.59%)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Predicted Label", fontweight="bold")
ax1.set_ylabel("True Label", fontweight="bold")
ax1.set_xticks([0, 1], ["negative", "positive"])
ax1.set_yticks([0, 1], ["negative", "positive"])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{cm_baseline[i, j]:,}", ha="center", va="center", color="white" if cm_baseline[i, j] > 6000 else "black", fontweight="bold")

# Stacked BiGRU confusion matrix:
# Test Acc 85.34%: Total mistakes 3666. TN: 10885, FP: 1615, FN: 2051, TP: 10449
cm_best = np.array([[10885, 1615], [2051, 10449]])
im2 = ax2.imshow(cm_best, cmap="Greens", interpolation="nearest")
ax2.set_title("Stacked BiGRU (Acc: 85.34%)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Predicted Label", fontweight="bold")
ax2.set_ylabel("True Label", fontweight="bold")
ax2.set_xticks([0, 1], ["negative", "positive"])
ax2.set_yticks([0, 1], ["negative", "positive"])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{cm_best[i, j]:,}", ha="center", va="center", color="white" if cm_best[i, j] > 6000 else "black", fontweight="bold")

plt.suptitle("Confusion Matrix Comparison (IMDB Test Set — 25,000 samples)", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
cm_comp_path = FIGURES_DIR / "confusion_matrix_comparison.png"
plt.savefig(cm_comp_path, dpi=160)
plt.close()
print(f"Generated {cm_comp_path}")
