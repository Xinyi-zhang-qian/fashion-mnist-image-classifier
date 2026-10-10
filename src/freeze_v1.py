"""复制已选候选为 V1，保存来源、配置和哈希；不训练、不评估测试集。"""

import hashlib
import json
import shutil

import torch

from data import PROJECT_DIR, OUTPUT_DIR


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = OUTPUT_DIR / "experiments/cnn_learning_rate_comparison/candidate_model.json"
    candidate = json.loads(source.read_text(encoding="utf-8"))
    destination = OUTPUT_DIR / "releases/v1"
    manifest_path = destination / "release_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, metadata in manifest["copied_files"].items():
            assert sha256(destination / name) == metadata["sha256"]
        assert sha256(destination / manifest["split_indices"]) == manifest["split_indices_sha256"]
        assert sha256(destination / "best_cnn.pt") == candidate["checkpoint_sha256"]
        print("V1 已存在，所有复制文件哈希核对通过；没有覆盖。")
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("V1 目录已有内容且没有完整清单，请先检查；不会覆盖。")
    weight = PROJECT_DIR / candidate["checkpoint"]
    history_path = PROJECT_DIR / candidate["source_history"]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    metrics_path = weight.parent / "cnn_class_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert candidate["learning_rate"] == 0.03 and candidate["best_epoch"] == 14
    assert sha256(weight) == candidate["checkpoint_sha256"] == metrics["checkpoint_sha256"]
    assert history["epochs"][13]["validation"] == candidate["validation_metrics"]
    assert metrics["accuracy"] == candidate["validation_metrics"]["accuracy"]
    assert not candidate["test_set_used"]
    for name in ("cnn_model.py", "data.py", "train.py"):
        assert sha256(PROJECT_DIR / "src" / name) == history["source_sha256"][name]
    destination.mkdir(parents=True, exist_ok=True)
    sources = {"best_cnn.pt": weight, "candidate_model.json": source,
               "cnn_training_history.json": history_path, "cnn_class_metrics.json": metrics_path,
               **{f"source/{name}": PROJECT_DIR / "src" / name
                  for name in ("cnn_model.py", "data.py", "train.py", "analyze_errors.py")}}
    copied = {}
    for name, original in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        assert sha256(target) == sha256(original)
        copied[name] = {"source": str(original), "sha256": sha256(target)}
    # 保存精确索引，不读取图片或官方测试集；与原 random_split 的实现相同。
    indices = torch.randperm(60_000, generator=torch.Generator().manual_seed(42)).tolist()
    predictions = json.loads((weight.parent / "cnn_validation_predictions.json").read_text())
    assert indices[50_000:] == [r["official_training_index"] for r in predictions["samples"]]
    split_path = destination / "split_indices.json"
    split_path.write_text(json.dumps({"split_seed": 42, "train": indices[:50_000],
                                     "validation": indices[50_000:]}) + "\n")
    manifest = {
        "release": "V1", "immutable_by_v2": True, "model": "best_cnn.pt",
        "checkpoint_sha256": sha256(weight), "original_checkpoint": str(weight),
        "original_candidate": str(source), "checkpoint_epoch": 14,
        "training_config": {**candidate["config"], "learning_rate": 0.03,
                            "optimizer_settings": {"momentum": 0, "dampening": 0,
                                "weight_decay": 0, "nesterov": False, "maximize": False,
                                "foreach": None, "differentiable": False},
                            "train_shuffle": True, "validation_shuffle": False,
                            "drop_last": False, "scheduler": None, "augmentation": None},
        "validation_metrics": candidate["validation_metrics"],
        "shirt_to_pullover": candidate["shirt_to_pullover"],
        "split_indices": "split_indices.json", "split_indices_sha256": sha256(split_path),
        "training_data_sha256": {name: sha256(PROJECT_DIR / "data/FashionMNIST/raw" / name)
                                 for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")},
        "test_status": "测试未进行", "test_set_used": False,
        "copied_files": copied,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"V1 冻结完成：{destination}\n权重 SHA-256：{sha256(weight)}\n测试未进行。")


if __name__ == "__main__":
    main()
