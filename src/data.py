"""训练前的数据准备：复用 FashionMNIST、转换图片、划分数据、创建批次。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import ToTensor


# 延续第一阶段的路径约定：所有数据和输出都放在当前项目内。
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

SEED = 42
TRAIN_SIZE = 50_000
VALIDATION_SIZE = 10_000
BATCH_SIZE = 64
NUM_WORKERS = 0

# 复用第一阶段的数据作者 HTTPS 下载地址。
DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
)


def ensure_fashion_mnist(include_test=True):
    """逐个补充缺失文件；训练演示可选择不检查或下载测试文件。"""
    raw_dir = DATA_DIR / "FashionMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # resources 包含训练/测试图片与标签的文件名，以及压缩文件的 MD5。
    for filename, md5 in FashionMNIST.resources:
        if not include_test and not filename.startswith("train-"):
            continue
        raw_path = raw_dir / Path(filename).stem  # 去掉 .gz 扩展名。
        if raw_path.is_file():
            print(f"复用本地文件：{raw_path.name}")
            continue

        print(f"补充缺失文件：{raw_path.name}")
        # 如果已有完整的 .gz，torchvision 会直接解压；否则才下载该文件。
        download_and_extract_archive(
            url=DOWNLOAD_URL + filename,
            download_root=str(raw_dir),
            filename=filename,
            md5=md5,
        )


def _load_training_split():
    """共享的官方训练集读取与划分逻辑，不读取官方测试集。"""
    transform = ToTensor()  # 灰度图片 -> [1, 28, 28] 的浮点张量，数值缩放到 [0, 1]。

    official_train = FashionMNIST(
        root=str(DATA_DIR), train=True, transform=transform, download=False
    )
    # 固定种子让每次调用得到相同的训练/验证索引。
    # random_split 返回两个 Subset：共享原始数据，只保存各自的样本索引。
    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, validation_dataset = random_split(
        official_train,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=split_generator,
    )
    return train_dataset, validation_dataset


def load_training_datasets():
    """只准备训练和验证部分，供不使用测试集的训练演示调用。"""
    ensure_fashion_mnist(include_test=False)
    return _load_training_split()


def load_datasets():
    """保留原接口：返回训练部分、验证部分和独立官方测试集。"""
    ensure_fashion_mnist()
    train_dataset, validation_dataset = _load_training_split()
    test_dataset = FashionMNIST(
        root=str(DATA_DIR), train=False, transform=ToTensor(), download=False
    )
    return train_dataset, validation_dataset, test_dataset


def create_train_loader(train_dataset):
    """复用同一套训练批次配置，不创建验证或测试 DataLoader。"""
    # 划分和批次使用独立的随机数生成器，避免两者互相影响。
    # 新建 loader 时可复现顺序；同一 loader 的每轮遍历仍会重新打乱。
    batch_generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=batch_generator,
    )


def create_validation_loader(validation_dataset):
    """复用验证批次配置，保持顺序，不需要加载测试集。"""
    return DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


def create_data_loaders(train_dataset, validation_dataset, test_dataset):
    """保留原接口：训练打乱，验证和测试保持顺序，每批最多 64 张。"""
    train_loader = create_train_loader(train_dataset)
    validation_loader = create_validation_loader(validation_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, validation_loader, test_loader
