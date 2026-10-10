"""加载固定最终CNN，预测训练样本或一张28×28灰度图片；不评估测试集。"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from data import OUTPUT_DIR, PROJECT_DIR, load_training_datasets
from final_model import load_final_model

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "final_model")
    parser.add_argument("--index", type=int, default=0, help="固定训练子集中的起始索引。")
    parser.add_argument("--count", type=int, default=9, help="展示1—64张训练样本。")
    parser.add_argument("--image", type=Path, help="可选：已有28×28图片路径，不下载数据。")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "final_prediction.png")
    args = parser.parse_args()
    if not 1 <= args.count <= 64:
        parser.error("count必须在1—64之间。")
    model, config = load_final_model((PROJECT_DIR / args.model_dir).resolve())
    names = config["class_names"]
    weight_path = (PROJECT_DIR / args.model_dir).resolve() / config["weights_file"]
    weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    if args.image is not None:
        # 自定义输入必须已经是28×28；不额外缩放或归一化，以保持ToTensor一致。
        with Image.open(args.image) as source:
            image = source.convert("L")
            if image.size != (28, 28):
                parser.error("自定义图片必须为28×28；真实照片不属于已验证适用范围。")
            images = ToTensor()(image).unsqueeze(0)
        labels = [None]
        identifiers = ["Custom image"]
    else:
        training, _ = load_training_datasets()  # 缺少训练数据时自动下载，未加载官方测试集。
        if args.index < 0 or args.index + args.count > len(training):
            parser.error("样本索引超出训练子集范围。")
        samples = [training[i] for i in range(args.index, args.index + args.count)]
        images = torch.stack([image for image, _ in samples])
        labels = [label for _, label in samples]
        identifiers = [f"Training subset {i}" for i in range(args.index, args.index + args.count)]

    assert images.dtype == torch.float32 and images.shape[1:] == (1, 28, 28)
    assert 0 <= images.min().item() <= images.max().item() <= 1
    with torch.inference_mode():
        scores = model(images)
        assert torch.isfinite(scores).all().item() and not scores.requires_grad
        predicted = scores.argmax(dim=1).tolist()
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == weight_hash

    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, image, label, prediction, identifier in zip(axes.flat, images, labels, predicted, identifiers):
        truth = "True: unknown" if label is None else f"True: {names[label]} ({label})"
        axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"{identifier}\n{truth}\nPred: {names[prediction]} ({prediction})",
                       color="black" if label is None else ("#287d3c" if label == prediction else "#b72d2d"),
                       fontsize=10)
        print(f"{identifier}: {truth}; predicted {prediction}: {names[prediction]}")
    figure.suptitle("Frozen final CNN - epoch 52\nSample inference only; not a new test-set evaluation")
    figure.tight_layout(rect=(0, 0, 1, .93))
    output = (PROJECT_DIR / args.output).resolve()
    if not output.is_relative_to(OUTPUT_DIR.resolve()):
        parser.error("输出图片必须位于本项目outputs内。")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white", transparent=False)
    plt.close(figure)
    print(f"权重SHA-256：{weight_hash}\n预测图片：{output}")
    print("样本演示完成，参数未更新；这些样本不能代替独立测试成绩。")


if __name__ == "__main__":
    main()
