"""检查训练前的数据准备，并保存第一个训练批次中的前 16 张图片。"""

import os
import sys

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from data import (
    BATCH_SIZE,
    DATA_DIR,
    NUM_WORKERS,
    OUTPUT_DIR,
    PROJECT_DIR,
    SEED,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    create_data_loaders,
    load_datasets,
)


# 延续第一阶段的绘图方式，缓存和图片保存在项目内。
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    print(f"Python 解释器：{sys.executable}")
    print(f"项目目录：{PROJECT_DIR}")
    print(f"数据目录：{DATA_DIR}")
    print(f"随机种子：{SEED}")
    train_dataset, validation_dataset, test_dataset = load_datasets()

    print(f"训练集数量：{len(train_dataset)}")
    print(f"验证集数量：{len(validation_dataset)}")
    print(f"官方测试集数量：{len(test_dataset)}")
    assert len(train_dataset) == TRAIN_SIZE
    assert len(validation_dataset) == VALIDATION_SIZE
    assert len(test_dataset) == 10_000

    # 不仅检查无重叠，还检查两部分覆盖全部官方训练样本。
    train_indices = set(train_dataset.indices)
    validation_indices = set(validation_dataset.indices)
    overlap = train_indices & validation_indices
    assert train_dataset.dataset is validation_dataset.dataset
    assert len(train_indices) == TRAIN_SIZE
    assert len(validation_indices) == VALIDATION_SIZE
    assert not overlap, "训练集与验证集出现索引重叠。"
    assert train_indices | validation_indices == set(range(TRAIN_SIZE + VALIDATION_SIZE))
    print(f"训练/验证索引交集：{len(overlap)}，完整覆盖 60,000 张官方训练图片。")

    # 一个样本：一张图片张量 + 一个真实类别编号。
    image, label = train_dataset[0]
    assert list(image.shape) == [1, 28, 28]
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all().item()
    assert 0 <= image.min().item() <= image.max().item() <= 1
    print(f"单张图片形状：{list(image.shape)}")
    print(f"单张图片类型：{image.dtype}")
    print(f"单张像素范围：[{image.min().item():.4f}, {image.max().item():.4f}]")
    print(f"单张真实标签：{label} -> {test_dataset.classes[label]}")

    train_loader, validation_loader, test_loader = create_data_loaders(
        train_dataset, validation_dataset, test_dataset
    )
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    for loader in (train_loader, validation_loader, test_loader):
        assert loader.batch_size == BATCH_SIZE
        assert loader.num_workers == NUM_WORKERS
        assert not loader.drop_last
    print(f"DataLoader：batch_size={BATCH_SIZE}，num_workers={NUM_WORKERS}")
    print("shuffle：训练=True，验证=False，测试=False")

    # iter() 开始遍历，next() 取出第一批；这里不会创建或训练模型。
    images, labels = next(iter(train_loader))
    assert list(images.shape) == [BATCH_SIZE, 1, 28, 28]
    assert list(labels.shape) == [BATCH_SIZE]
    assert images.dtype == torch.float32 and labels.dtype == torch.int64
    assert torch.isfinite(images).all().item()
    assert 0 <= images.min().item() <= images.max().item() <= 1
    assert 0 <= labels.min().item() <= labels.max().item() < len(test_dataset.classes)
    print(f"第一批图片形状：{list(images.shape)}，类型：{images.dtype}")
    print(f"第一批标签形状：{list(labels.shape)}，类型：{labels.dtype}")
    print(f"第一批像素范围：[{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"展示的前 16 个真实标签：{labels[:16].tolist()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    for index, axis in enumerate(axes.flatten()):
        label = labels[index].item()
        class_name = test_dataset.classes[label]
        # 去掉大小为 1 的通道维度，使 Matplotlib 接收 [28, 28] 的图片。
        axis.imshow(
            images[index].squeeze(0).numpy(),
            cmap="gray", vmin=0, vmax=1, interpolation="nearest",
        )
        axis.set_title(f"Batch item {index}\nLabel {label}: {class_name}")
        axis.axis("off")

    figure.suptitle("FashionMNIST: first 16 training-batch samples (true labels)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = OUTPUT_DIR / "batch_samples.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"批次样本图已保存：{output_path}")
    print("全部数据检查通过。")


if __name__ == "__main__":
    main()
