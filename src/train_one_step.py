"""CPU 上的新模型、一个训练批次、一次 SGD 更新；不保存模型参数。"""

import math
import os
import sys

import torch
from torch import nn

from data import OUTPUT_DIR, PROJECT_DIR, SEED, create_train_loader, load_training_datasets
from model import SimpleClassifier


LEARNING_RATE = 0.01
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def save_predictions(images, labels, predictions_before, class_names):
    """只绘图，展示已记录的训练前预测，不创建模型或更新参数。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    # 当前 Mac 已有此字体，用于显示用户要求的中文标题，无需安装。
    title_font = FontProperties(family="Arial Unicode MS", size=13)
    figure.suptitle(
        "训练前预测 / Pre-training predictions (random initialization)\n"
        "Model: one update only - these predictions are not training results",
        fontproperties=title_font,
    )
    for sample_index, axis in enumerate(axes.flatten()):
        true_label = labels[sample_index].item()
        predicted_label = predictions_before[sample_index].item()
        axis.imshow(
            images[sample_index].squeeze(0).numpy(),
            cmap="gray", vmin=0, vmax=1, interpolation="nearest",
        )
        axis.set_title(
            f"True: {true_label} - {class_names[true_label]}\n"
            f"Before: {predicted_label} - {class_names[predicted_label]}",
            fontsize=10,
        )
        axis.axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output_path = OUTPUT_DIR / "one_step_predictions.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    print(f"Python 解释器：{sys.executable}")
    print(f"项目目录：{PROJECT_DIR}")
    print(f"设备：{device}；随机种子：{SEED}；学习率：{LEARNING_RATE}")

    # 复用既有划分和批次逻辑，只加载官方训练集，不加载或评估测试集。
    train_dataset, _ = load_training_datasets()
    train_loader = create_train_loader(train_dataset)
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    class_names = train_dataset.dataset.classes
    print(f"图片形状：{list(images.shape)}；标签形状：{list(labels.shape)}")

    model = SimpleClassifier().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # detach() 不跟踪梯度，clone() 真正复制数据；副本不会随原参数更新。
    first_layer_weight = model.layers[1].weight
    weights_before = first_layer_weight.detach().clone()
    assert weights_before.data_ptr() != first_layer_weight.data_ptr()

    print("\n1. zero_grad：清除已有梯度，避免梯度累积。")
    optimizer.zero_grad()
    print(f"第一层权重的梯度：{first_layer_weight.grad}")

    print("\n2. 前向计算：用当前权重计算 10 个原始类别分数。")
    outputs = model(images)
    assert list(outputs.shape) == [images.shape[0], 10]
    assert outputs.requires_grad and torch.isfinite(outputs).all().item()
    predictions_before = outputs.detach().argmax(dim=1)
    assert torch.equal(first_layer_weight.detach(), weights_before)
    assert first_layer_weight.grad is None
    print(f"分数形状：{list(outputs.shape)}；第一层权重保持不变。")

    print("\n3. 损失：直接将原始分数和真实类别传入 CrossEntropyLoss，不提前做 Softmax。")
    loss = criterion(outputs, labels)
    assert loss.ndim == 0 and torch.isfinite(loss).item()
    loss_before = loss.item()
    assert torch.equal(first_layer_weight.detach(), weights_before)
    print(f"更新前的批次平均损失：{loss_before:.9f}")

    print("\n4. backward：计算损失对权重的梯度，此时不修改权重。")
    loss.backward()
    gradient = first_layer_weight.grad
    assert gradient is not None and torch.isfinite(gradient).all().item()
    assert torch.equal(first_layer_weight.detach(), weights_before)
    print(f"第一层梯度形状：{list(gradient.shape)}")
    print(f"第一层梯度 L2 范数：{gradient.norm().item():.9f}")
    print("检查通过：反向传播后，第一层全部权重与更新前副本完全相同。")

    # 选择绝对梯度最大的元素，确保观察到一个非零梯度，不重跑或挑选损失结果。
    flat_index = gradient.abs().argmax().item()
    row = flat_index // gradient.shape[1]
    column = flat_index % gradient.shape[1]
    index = (row, column)
    gradient_value = gradient[index].item()
    assert gradient_value != 0
    value_before = weights_before[index].item()
    value_after_backward = first_layer_weight[index].item()
    print(f"选定第一层权重索引：{index}（绝对梯度最大）")
    print(f"更新前值：{value_before:+.9f}")
    print(f"反向传播后值：{value_after_backward:+.9f}")
    print(f"该权重的梯度：{gradient_value:+.9f}")

    print("\n5. step：SGD 用梯度更新参数，新值 = 旧值 - 学习率 × 梯度。")
    optimizer.step()  # 只更新一次，没有训练循环。
    value_after = first_layer_weight[index].item()
    expected_value = value_before - LEARNING_RATE * gradient_value
    assert math.isclose(value_after, expected_value, rel_tol=1e-5, abs_tol=1e-8)
    assert value_after != value_before
    expected_weights = weights_before - LEARNING_RATE * gradient.detach()
    assert torch.allclose(first_layer_weight.detach(), expected_weights, rtol=1e-5, atol=1e-8)
    print(f"step 后值：{value_after:+.9f}")
    print(f"公式预期值：{expected_value:+.9f}")
    print(f"浮点误差：{abs(value_after - expected_value):.3e}")
    print("检查通过：step 才改变权重，整个第一层权重符合 SGD 更新公式。")

    # 只观察同一批次的新损失，不再次反向传播或更新参数。
    with torch.no_grad():
        loss_after = criterion(model(images), labels).item()
    assert math.isfinite(loss_after)
    print(f"\n同一批损失：{loss_before:.9f} -> {loss_after:.9f}")
    if loss_after < loss_before:
        print("本次观察：同一批次的损失下降。")
    elif loss_after > loss_before:
        print("本次观察：同一批次的损失上升，如实保留结果。")
    else:
        print("本次观察：同一批次的损失没有变化。")
    print("这只反映当前训练批次，不代表对新数据的表现；模型仅更新一次，预测尚不可靠。")

    output_path = save_predictions(images, labels, predictions_before, class_names)
    print(f"更新前的前 9 个预测类别：{predictions_before[:9].tolist()}")
    print(f"前 9 个真实类别：{labels[:9].tolist()}")
    print(f"图片已保存：{output_path}")
    print("演示完成：一个训练批次，一次更新，不保存模型参数。")


if __name__ == "__main__":
    main()
