"""简单全连接分类模型：本阶段只定义结构，不训练。"""

from torch import nn


class SimpleClassifier(nn.Module):
    """输入一批灰度图片，为每张图片输出 10 个原始类别分数。"""

    def __init__(self):
        super().__init__()
        # Sequential 按顺序执行各层；Flatten 保留最前面的批次维度。
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),  # [B, 1, 28, 28] -> [B, 784]
            nn.Linear(784, 128),      # [B, 784] -> [B, 128]
            nn.ReLU(),               # 负数变为 0，形状不变。
            nn.Linear(128, 10),       # [B, 128] -> [B, 10]，分数不是概率。
        )

    def forward(self, images):
        # 调用 model(images) 时，PyTorch 会执行这个方法。
        return self.layers(images)
