"""验证未训练模型的前向计算，并展示一张图片和 10 个原始分数。"""

import os
import sys

import torch

from data import OUTPUT_DIR, PROJECT_DIR, SEED, create_data_loaders, load_datasets
from model import SimpleClassifier


# 复用已有绘图路径约定，图片与 Matplotlib 缓存都放在本项目内。
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # 固定随机种子，让模型的随机初始参数可复现。
    torch.manual_seed(SEED)
    print(f"Python 解释器：{sys.executable}")
    print(f"项目目录：{PROJECT_DIR}")
    print(f"随机种子：{SEED}")

    # 复用数据准备阶段的划分和 DataLoader，不再实现一套加载逻辑。
    train_dataset, validation_dataset, test_dataset = load_datasets()
    train_loader, _, _ = create_data_loaders(
        train_dataset, validation_dataset, test_dataset
    )
    images, labels = next(iter(train_loader))
    batch_size = images.shape[0]  # 使用实际大小，不把模型限定为每批 64 张。
    class_names = train_dataset.dataset.classes
    print(f"图片形状：{list(images.shape)}")
    print(f"标签形状：{list(labels.shape)}")
    assert list(images.shape) == [batch_size, 1, 28, 28]
    assert list(labels.shape) == [batch_size]

    model = SimpleClassifier()
    model.eval()
    print(f"模型结构：\n{model}")

    # 复制当前参数，用于确认前向计算后参数没有变化。
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    # no_grad() 不记录用于反向传播的计算图；这里只计算分数。
    with torch.no_grad():
        scores = model(images)

        # 对同一批图片逐层观察，验证结果与完整的 model(images) 一致。
        features = images
        for layer in model.layers:
            features = layer(features)
            print(f"经过 {layer} 后：{list(features.shape)}")
        assert torch.equal(features, scores)

    assert list(scores.shape) == [batch_size, len(class_names)] == [batch_size, 10]
    assert torch.isfinite(scores).all().item()
    assert not scores.requires_grad
    for name, parameter in model.named_parameters():
        assert torch.equal(parameters_before[name], parameter.detach())
        assert parameter.grad is None
    print(f"模型输出形状：{list(scores.shape)}")
    print("参数保持不变，没有计算参数梯度。")

    # 一张图片对应一行分数，分数最大的类别就是当前预测类别。
    first_scores = scores[0]
    predicted_label = first_scores.argmax().item()
    true_label = labels[0].item()
    print("第一张图片的 10 个原始类别分数（不是概率）：")
    for index, score in enumerate(first_scores.tolist()):
        print(f"  {index}: {class_names[index]:<12} {score:.6f}")
    print(f"预测类别：{predicted_label} -> {class_names[predicted_label]}")
    print(f"真实类别：{true_label} -> {class_names[true_label]}")

    # numel() 是张量中的数值个数；这里只统计允许学习的参数。
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"可训练参数数量：{parameter_count:,}")
    print("这是未训练模型，输出来自随机初始化参数，不能据此判断识别能力。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, (image_axis, score_axis) = plt.subplots(1, 2, figsize=(13, 5))
    figure.suptitle("Untrained model - forward pass only", fontsize=16)

    image_axis.imshow(
        images[0].squeeze(0).numpy(),
        cmap="gray", vmin=0, vmax=1, interpolation="nearest",
    )
    image_axis.set_title(
        f"Input image\nTrue: {class_names[true_label]} ({true_label})\n"
        f"Predicted: {class_names[predicted_label]} ({predicted_label})"
    )
    image_axis.axis("off")

    # 柱状图直接画原始分数，不使用 Softmax 或百分比。
    colors = [
        "#e7833b" if index == predicted_label else "#4c7ac7"
        for index in range(len(class_names))
    ]
    bars = score_axis.barh(range(len(class_names)), first_scores.numpy(), color=colors)
    score_axis.set_yticks(
        range(len(class_names)),
        labels=[f"{index}: {name}" for index, name in enumerate(class_names)],
    )
    score_axis.invert_yaxis()
    score_axis.axvline(0, color="black", linewidth=0.8)
    score_axis.set_xlabel("Raw score (not probability)")
    score_axis.set_title("10 raw class scores (logits)\nOrange: predicted class")
    score_axis.bar_label(
        bars, labels=[f"{score:.3f}" for score in first_scores.tolist()], padding=3,
    )
    score_axis.margins(x=0.25)

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output_path = OUTPUT_DIR / "untrained_prediction.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"可视化已保存：{output_path}")
    print("全部模型前向检查通过。")


if __name__ == "__main__":
    main()
