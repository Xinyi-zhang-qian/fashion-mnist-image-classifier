"""加载发布版的固定模型，先核对配置与权重哈希。"""

import hashlib
import json
from pathlib import Path

import torch

from cnn_model import SimpleCNN
from data import PROJECT_DIR


def load_final_model(directory=None):
    directory = PROJECT_DIR / "final_model" if directory is None else Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    weight = directory / config["weights_file"]
    actual_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
    if actual_hash != config["weights_sha256"]:
        raise ValueError("最终模型权重哈希不匹配，请重新获取完整发布包。")
    if config["model"]["name"] != "SimpleCNN" or config["preprocessing"]["name"] != "ToTensor":
        raise ValueError("本入口仅支持记录中的SimpleCNN与ToTensor预处理。")
    torch.set_num_threads(config["training_config"]["cpu_threads"])
    model = SimpleCNN().to("cpu")
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    assert str(model) == config["model"]["structure"]
    assert sum(p.numel() for p in model.parameters()) == config["model"]["parameter_count"]
    model.eval()
    return model, config
