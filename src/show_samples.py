"""第一阶段：读取 FashionMNIST，保存前 9 张图片和它们的真实标签。"""

import os
from pathlib import Path


# 从脚本位置定位项目，避免因为运行命令的目录不同而把文件放错地方。
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# 将 Matplotlib 的缓存放在本项目内，适应当前工作区的写入权限。
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib

# 脚本只保存图片，之后用 Mac 的 open 命令打开图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST


def main():
    """准备数据，画出 3×3 样本图，并打印结果。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # torchvision 0.17.2 默认使用 HTTP；这里改用数据作者的 HTTPS 仓库。
    # torchvision 仍会校验下载文件的 MD5，确认数据完整。
    FashionMNIST.mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    ]

    print(f"项目目录：{PROJECT_DIR}")
    print(f"数据目录：{DATA_DIR}")
    print("正在读取 FashionMNIST；缺少数据时会自动下载。", flush=True)
    dataset = FashionMNIST(
        root=str(DATA_DIR),
        train=True,      # 选择训练集；这里只读取数据，不训练模型。
        download=True,   # 已有数据时直接复用。
    )

    print(f"训练集样本数：{len(dataset)}")
    print(f"单张图片尺寸：{dataset[0][0].size}，灰度图")
    print(f"类别数：{len(dataset.classes)}")

    # 创建 3 行、3 列的画布。flatten() 把九个子图变成一列，方便遍历。
    figure, axes = plt.subplots(3, 3, figsize=(8, 8))
    for index, axis in enumerate(axes.flatten()):
        image, label = dataset[index]  # 一个样本 = 图片 + 真实类别编号。
        class_name = dataset.classes[label]

        # 所有图片使用相同的灰度范围；nearest 保留原始像素的方格效果。
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"Sample {index}\nLabel {label}: {class_name}")
        axis.axis("off")
        print(f"样本 {index}：标签 {label} -> {class_name}")

    figure.suptitle("FashionMNIST: 9 samples with true labels", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = OUTPUT_DIR / "fashion_samples.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"样本图已保存：{output_path}")


# 直接运行这个文件时才执行 main()。
if __name__ == "__main__":
    main()
