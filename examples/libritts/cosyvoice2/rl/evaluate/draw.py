import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 配置参数
result_dir = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/rl/evaluate/result"
output_dir = "./visualizations"
os.makedirs(output_dir, exist_ok=True)

# 自动生成颜色映射
colors = list(mcolors.TABLEAU_COLORS.values())

# 数据结构初始化
data = {
    "accuracy": {"dpo": {}, "emo_dpo": {}, "init": {}},
    "emoSIM": {"dpo": {}, "emo_dpo": {}, "init": {}},
    "proSIM": {"dpo": {}, "emo_dpo": {}, "init": {}},
    "wer": {"dpo": {}, "emo_dpo": {}, "init": {}}
}

# 解析文件名并加载数据
for filename in os.listdir(result_dir):
    if not filename.endswith(".txt"):
        continue
    
    # 解析文件名
    parts = filename[:-4].split("_")
    metric = parts[1]
    method = "init" if "init" in parts else "emo_dpo" if "emo" in parts and "dpo" in parts else parts[3]
    
    # 获取epoch信息
    epoch = 0 if "init" in parts else int(parts[-1]) if "epoch" in parts else 0

    # 读取文件内容
    with open(os.path.join(result_dir, filename)) as f:
        content = f.read().strip()

    # 处理不同指标的数据
    if metric == "accuracy":
        for line in content.split("\n"):
            emotion, value = line.split(" acc: ")
            key = f"accuracy_{emotion.strip()}"
            if key not in data:
                data[key] = {"dpo": {}, "emo_dpo": {}, "init": {}}
            data[key][method].setdefault(epoch, float(value))
    else:
        value = float(content.split(": ")[1])
        data[metric][method].setdefault(epoch, value)

# 1. 绘制折线图
for metric in data:
    plt.figure(figsize=(10, 6))
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    
    for idx, method in enumerate(["dpo", "emo_dpo", "init"]):
        epochs = sorted(data[metric][method].keys())
        values = [data[metric][method][e] for e in epochs]
        
        if method == "init":
            plt.scatter(epochs, values, color=colors[idx], s=100, label=method)
        else:
            plt.plot(epochs, values, "o-", color=colors[idx], label=method)
    
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"line_{metric}.png"))
    plt.close()

# 2. 最终epoch对比柱状图（使用emoSIM作为示例）
# 2. 最终epoch对比柱状图（在指标名称旁添加箭头）
final_results = {
    "dpo": {
        "emoSIM": data["emoSIM"]["dpo"][9],
        "proSIM": data["proSIM"]["dpo"][9],
        "WER": data["wer"]["dpo"][9]
    },
    "emo_dpo": {
        "emoSIM": data["emoSIM"]["emo_dpo"][9],
        "proSIM": data["proSIM"]["emo_dpo"][9],
        "WER": data["wer"]["emo_dpo"][9]
    },
    "init": {
        "emoSIM": data["emoSIM"]["init"][0],
        "proSIM": data["proSIM"]["init"][0],
        "WER": data["wer"]["init"][0]
    }
}

metrics = ["emoSIM", "proSIM", "WER"]
methods = ["dpo", "emo_dpo", "init"]

plt.figure(figsize=(12, 6))
bar_width = 0.25
x = range(len(metrics))

# 定义每个指标的箭头方向（1表示上箭头，-1表示下箭头）
arrow_directions = {
    "emoSIM": 1,    # 越大越好
    "proSIM": 1,    # 越大越好
    "WER": -1       # 越小越好
}

# 绘制柱状图
for idx, method in enumerate(methods):
    values = [final_results[method][m] for m in metrics]
    plt.bar([i + bar_width*idx for i in x], values, bar_width, label=method, color=colors[idx])

# 修改x轴标签，添加箭头
xtick_labels = []
for metric in metrics:
    arrow = "↑" if arrow_directions[metric] == 1 else "↓"
    xtick_labels.append(f"{metric}\n({arrow})")

plt.title("Final Epoch Comparison (emoSIM, proSIM, WER)")
plt.xticks([i + bar_width for i in x], xtick_labels)
plt.ylabel("Value")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_epoch_comparison.png"))
plt.close()


print(f"Visualizations saved to: {output_dir}")
