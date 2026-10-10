"""用内存中的临时模型和一个批次，演示损失、反向传播及一次 SGD 更新。"""

import argparse
import math
import sys

import torch
from torch import nn

from data import SEED, create_data_loaders, load_datasets
from model import SimpleClassifier


LEARNING_RATE = 0.1
TRACKED_INDEX = (8, 0)  # 最后一层：隐藏特征 0 -> 类别 8（Bag）。


def explain_step(number, explanation, step_by_step):
    """每步先讲解；逐步模式下，按回车才执行这一小步。"""
    print(f"\n第 {number} 步：{explanation}", flush=True)
    if step_by_step:
        input("按回车执行这一步：")


def parameters_unchanged(model, snapshot):
    """比较全部参数，避免只看一个权重而漏掉其他变化。"""
    return all(
        torch.equal(parameter.detach(), snapshot[name])
        for name, parameter in model.named_parameters()
    )


def print_weight(stage, parameter):
    """读取权重及其对应的梯度；梯度保存在整个参数张量的 .grad 中。"""
    value = parameter[TRACKED_INDEX].item()
    gradient = (
        None if parameter.grad is None else parameter.grad[TRACKED_INDEX].item()
    )
    gradient_text = "None" if gradient is None else f"{gradient:+.9f}"
    print(f"{stage}：weight[8, 0] = {value:+.9f}；梯度 = {gradient_text}", flush=True)
    return value, gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-by-step", action="store_true", help="每一步讲解后，等待按回车再执行。"
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    print(f"Python 解释器：{sys.executable}")
    print(f"随机种子：{SEED}；SGD 学习率：{LEARNING_RATE}")
    datasets = load_datasets()
    train_loader, _, _ = create_data_loaders(*datasets)
    images, labels = next(iter(train_loader))  # 只取一个批次。
    print(f"图片形状：{list(images.shape)}；标签形状：{list(labels.shape)}")

    # 新建模型只存在于本次 Python 进程中，不读取或保存任何模型参数文件。
    model = SimpleClassifier()
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)  # 清除梯度；新模型的梯度本来就是 None。
    tracked_parameter = model.layers[3].weight
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    print("临时模型已准备；跟踪最后一层 weight[8, 0]，尚未更新。")
    print_weight("初始状态", tracked_parameter)

    explain_step(
        1,
        "前向计算：用当前参数得到每张图片的 10 个分数。"
        "这次保留计算图，供稍后的反向传播使用；不使用 no_grad()。",
        args.step_by_step,
    )
    scores = model(images)
    assert list(scores.shape) == [images.shape[0], 10]
    assert scores.requires_grad
    assert parameters_unchanged(model, initial_parameters)
    assert all(parameter.grad is None for parameter in model.parameters())
    print(f"输出形状：{list(scores.shape)}；requires_grad={scores.requires_grad}")
    print_weight("前向计算后", tracked_parameter)
    print("检查通过：前向计算未改变参数，也未生成参数梯度。")

    explain_step(
        2,
        "损失计算：CrossEntropyLoss 接收原始分数与真实类别编号，"
        "衡量预测和标签的差距，返回本批次的平均损失。无需先做 Softmax。",
        args.step_by_step,
    )
    loss = loss_function(scores, labels)
    assert loss.ndim == 0 and loss.requires_grad
    assert torch.isfinite(loss).item()
    assert parameters_unchanged(model, initial_parameters)
    assert all(parameter.grad is None for parameter in model.parameters())
    loss_before = loss.item()
    print(f"批次平均损失：{loss_before:.9f}；形状：{list(loss.shape)}（标量）")
    value_before, gradient_before = print_weight("反向传播前", tracked_parameter)
    assert gradient_before is None
    print("检查通过：计算损失仍未改变参数，也未生成参数梯度。")

    explain_step(
        3,
        "反向传播：loss.backward() 沿计算图求出损失对各参数的梯度。"
        "梯度描述参数小幅变化对损失的局部影响；这一步不修改权重。",
        args.step_by_step,
    )
    loss.backward()
    assert parameters_unchanged(model, initial_parameters)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    value_after_backward, gradient = print_weight("反向传播后", tracked_parameter)
    assert value_after_backward == value_before
    assert gradient is not None and gradient != 0
    print("检查通过：全部参数完全不变，现在有了梯度。")

    explain_step(
        4,
        "优化器更新：optimizer.step() 使用梯度修改参数。"
        "本次 SGD 没有动量或权重衰减，公式为 新权重 = 旧权重 - 学习率 × 梯度。",
        args.step_by_step,
    )
    optimizer.step()  # 整个脚本只有这一次参数更新，没有训练循环。
    value_after_step, gradient_after_step = print_weight("optimizer.step() 后", tracked_parameter)
    expected_value = value_before - LEARNING_RATE * gradient
    assert value_after_step != value_before
    assert not parameters_unchanged(model, initial_parameters)
    assert math.isclose(value_after_step, expected_value, rel_tol=1e-6, abs_tol=1e-8)
    assert gradient_after_step == gradient  # step() 不会自动清空梯度。
    print(f"按公式计算的新权重：{expected_value:+.9f}")
    print(f"实际权重变化：{value_after_step - value_before:+.9f}")
    print("检查通过：step 才改变参数，数值符合 SGD 更新公式。")
    print("step 不会自动清空梯度；下次更新前需要 zero_grad()。")

    # 只观察同一批次更新后的损失，不反向传播、不再次更新。
    with torch.no_grad():
        loss_after = loss_function(model(images), labels).item()
    print(f"同一批次损失：更新前 {loss_before:.9f} -> 更新后 {loss_after:.9f}")
    print("这只是本批次此次更新的效果，不代表完整模型的识别能力。")
    print("演示结束：一个批次，一次更新；临时模型未保存，已有文件不受影响。")


if __name__ == "__main__":
    main()
