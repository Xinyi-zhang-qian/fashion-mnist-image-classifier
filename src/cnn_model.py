"""小型 CNN：两次卷积和池化，然后输出 10 个原始类别分数。"""

from torch import nn


class SimpleCNN(nn.Module):
    """保留图片的二维结构，从局部特征逐步得到分类分数。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                           # [B, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [B, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                           # [B, 32, 7, 7]
            nn.Flatten(start_dim=1),                   # [B, 1568]
            nn.Linear(32 * 7 * 7, 64),                 # [B, 64]
            nn.ReLU(),
            nn.Linear(64, 10),                         # [B, 10]，原始分数。
        )

    def forward(self, images):
        return self.layers(images)
