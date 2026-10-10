"""只检查 CNN 每层形状、输出和参数数量，不更新参数。"""

import sys

import torch

from cnn_model import SimpleCNN
from data import SEED, create_train_loader, load_training_datasets


def check_shapes(train_dataset, device):
    """使用原训练批次，逐层检查；该临时模型不会用于正式训练。"""
    torch.manual_seed(SEED)
    images, labels = next(iter(create_train_loader(train_dataset)))
    images = images.to(device)
    batch_size = len(labels)
    model = SimpleCNN().to(device)
    model.eval()
    before = [p.detach().clone() for p in model.parameters()]
    expected = [
        [batch_size, 16, 28, 28], [batch_size, 16, 28, 28],
        [batch_size, 16, 14, 14],
        [batch_size, 32, 14, 14], [batch_size, 32, 14, 14],
        [batch_size, 32, 7, 7],
        [batch_size, 1568], [batch_size, 64], [batch_size, 64],
        [batch_size, 10],
    ]
    assert list(images.shape) == [batch_size, 1, 28, 28]
    print(f"\nCNN 形状检查：图片 {list(images.shape)}，标签 {list(labels.shape)}。")
    shapes = [list(images.shape)]
    with torch.no_grad():
        features = images
        for layer, shape in zip(model.layers, expected):
            features = layer(features)
            assert list(features.shape) == shape
            shapes.append(list(features.shape))
            print(f"  {layer}: {list(features.shape)}")
        assert torch.equal(features, model(images))
        assert torch.isfinite(features).all().item()
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), original)
        assert parameter.grad is None
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总数：{parameter_count:,}；前向检查没有改变参数。", flush=True)
    return {"batch_size": batch_size, "layer_shapes": shapes, "parameter_count": parameter_count}


def main():
    print(f"Python 解释器：{sys.executable}")
    train_dataset, _ = load_training_datasets()
    check_shapes(train_dataset, torch.device("cpu"))


if __name__ == "__main__":
    main()
