import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

DIRECTORY = "./temps"
if not os.path.isdir(DIRECTORY):
    exit()

def get_file_data(dir: str, threshold: float):
    metrics = []
    for file in sorted(os.listdir(dir)):
        file_name = file.replace(".csv", "")
        file_name_list = file_name.split("_")
        # if "inception" in file_name_list:
        #     continue
        if file.endswith(".csv") and f"{int(threshold*100)}" == file_name_list[3]:
            df = pd.read_csv(os.path.join(dir, file), index_col=0)
            accuracy = df.loc["accuracy"]["precision"]
            recall = df.loc["macro avg"]["recall"]
            f1_score = df.loc["macro avg"]["f1-score"]
            # model_name = " ".join([word for word in file_name_list if word != f"{int(threshold * 100)}"])
            metrics.append((file_name, accuracy, recall, f1_score))
    return metrics

# metrics_50 = get_file_data(DIRECTORY, 0.5)
# metrics_80 = get_file_data(DIRECTORY, 0.8)
metrics_90 = get_file_data(DIRECTORY, 0.9)
metrics_100 = get_file_data(DIRECTORY, 1.0)
metrics = sorted(metrics_90 + metrics_100)


model_names = [m[0] for m in metrics]
accuracy_vals = [m[1] for m in metrics]
recall_vals = [m[2] for m in metrics]
f1_vals = [m[3] for m in metrics]

# Metric display order
metric_names = ["Accuracy", "Macro Recall", "Macro F1"]
metric_colors = {
    "Accuracy": "steelblue",
    "Macro Recall": "seagreen",
    "Macro F1": "indianred"
}
metrics_data = [accuracy_vals, recall_vals, f1_vals]

# Plotting config
x = np.arange(len(model_names))  # group positions
bar_width = 0.2

fig, ax = plt.subplots(figsize=(30, 18))
for i, (metric, values) in enumerate(zip(metric_names, metrics_data)):
    bar_pos = x + (i - 1) * bar_width  # offset - center bars
    bars = ax.bar(bar_pos, values, width=bar_width, label=metric, color=metric_colors[metric])

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=18)

shortened_names = [
    name.replace("mobilenet_v3_large_", "mbv3\ndom=").replace("_global_prune_", "\nglobal prune=")
    if "mobilenet_v3_large" in name
    else name.replace("inception_v3_299_", "iv3\ndom=").replace("_prune_", "\nprune=")
    for name in model_names
]

section_breaks = [1]  # Insert red line *before* these indices
for i in section_breaks:
    ax.axvline(x=i - 0.5, color='red', linestyle='--', linewidth=3)
# Final styling
ax.set_xticks(x)
ax.set_xticklabels(shortened_names, rotation=0, ha='center', fontsize=16)
# ax.set_xticklabels(shortened_names, rotation=0), ha
ax.set_ylabel("Score")
# ax.set_title("Model Comparison by Accuracy, F1, Recall; MobileNetV3: 15 Epoch; Prune per layer", fontsize=32)
ax.set_title("Model Comparison by Accuracy, F1, Recall: InceptionV3: 30 Epoch, MobileNetV3: 15 Epoch", fontsize=32)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=30)
plt.tight_layout()
plt.savefig("evaluation_plot.png")