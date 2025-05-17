import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
# 配置参数
result_dir = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/rl/evaluate/result"

output_dir = Path(__file__).parent / "visualizations"
os.makedirs(output_dir, exist_ok=True)

def visualize_epoch(train_corpus, test_corpus):
    # 自动生成颜色映射
    colors = list(mcolors.TABLEAU_COLORS.values())

    # 数据结构初始化
    data = {}

    # 解析文件名并加载数据
    for filename in os.listdir(result_dir):
        full_path = os.path.join(result_dir, filename)
        if os.path.isdir(full_path):
            continue
        
        # 解析文件名
        file_len = len(filename.split('_')) 
        if file_len== 4:
            continue
        # {test_corpus}_{${train_corpus_name}_${method}|b${beta}c${clip}s${emo_train_start_epoch}l${lr}}_epoch_{num_epoch} like
        elif file_len == 5:
            splits = filename.split('_')
            f_test_corpus = splits[0]
            f_train_corpus = splits[1]
            if f_test_corpus != test_corpus or f_train_corpus != train_corpus:
                continue
            method_hyp = filename.split('_')[2]
            if '|' in method_hyp and 'l' in method_hyp and 's' in method_hyp:
                method = method_hyp.split('|')[0]
                hyp = method_hyp.split('|')[1]
                start_epoch = int(hyp.split('s')[1].split('l')[0])

            epoch = filename.split('_')[-1]
            
            if "emo-grpo" in filename:
                epoch = int(epoch) + start_epoch + 1
                epoch = str(epoch)
                print(f"INFO: 基于start_epoch {start_epoch}，更新emo-grpo的epoch")
            print(f"加载{filename}")


        # 读取文件内容
        with open(os.path.join(result_dir, filename)) as f:
            content = f.read().strip()
        if len(content.split("\n")) > 9:
            raise Exception("文件内容超过9行")
        for line in content.split("\n"):
            metric, value = line.split(': ')
            if 'acc' in metric:
                emotion = metric.split()[0]
                # metric_emotion
                key = f"accuracy_{emotion.strip()}"
                if key not in data:
                    data[key] = {}
                if method not in data[key]:
                    data[key][method] = {}
                data[key][method][epoch] = float(value)
            else:
                if metric not in data:
                    data[metric] = {}
                if method not in data[metric]:
                    data[metric][method] = {}
                data[metric][method][epoch] = float(value)

    # 绘制折线图
    for metric in data:
        plt.figure(figsize=(10, 6))
        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        all_epochs = sorted({int(e) for method in data[metric].values() for e in method.keys()})

        # 排除
        data[metric] = {k.replace("PS2", "").replace("nlnl", "r2").split('|')[0]: v for k, v in data[metric].items() 
                        if "DPO|beta0.01" not in k and "EmoDPO|beta0.1" not in k
                        and "olnl" not in k}
        sorted_methods = sorted(data[metric].keys())
        for idx, method in enumerate(sorted_methods):
           # 获取当前方法的所有 epoch（转换为整数并排序）
            method_epochs = sorted(map(int, data[metric][method].keys()))
            method_values = [data[metric][method][str(e)] for e in method_epochs]

            if len(method_epochs) == 1:
                plt.axhline(y=method_values[0], color=colors[idx], linestyle='--', label=method, alpha=0.7)
                plt.scatter(method_epochs, method_values, color=colors[idx], s=100, zorder=3)
            else:
                plt.plot(method_epochs, method_values, "o-", color=colors[idx], label=method)
        
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{test_corpus}_{train_corpus}_line_epoch_{metric}.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化，目前只支持同一训练集和测试集的不同方法的可视化")
    parser.add_argument("--train_corpus", required=True, 
                        help="使用的train数据集")
    parser.add_argument("--test_corpus", required=True, 
                        help="使用的test数据集")
    args = parser.parse_args()
    visualize_epoch(args.train_corpus, args.test_corpus)