"""核对已有记录并整理最终推理模型；不读取数据集或执行模型推理。"""

import hashlib
import json
from pathlib import Path
import shutil

import torch

from data import PROJECT_DIR


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = PROJECT_DIR / "outputs/experiments/v2_resume45_lr001_15epochs"
    evaluation = PROJECT_DIR / "outputs/final_evaluation"
    history = json.loads((source / "cnn_training_history.json").read_text(encoding="utf-8"))
    manifest = json.loads((evaluation / "model_manifest.json").read_text(encoding="utf-8"))
    test = json.loads((evaluation / "test_metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(source / "checkpoint_epoch_52.pt", map_location="cpu", weights_only=False)
    weights = torch.load(source / "best_cnn.pt", map_location="cpu", weights_only=True)
    digest = sha256(source / "best_cnn.pt")
    assert history["best_epoch"] == manifest["selected_epoch"] == checkpoint["completed_epoch"] == 52
    assert digest == history["checkpoint_sha256"] == manifest["weight_sha256"] == test["checkpoint_sha256"]
    assert digest == "c972ca96c5d63c81998a6d884f57ebc42406f04a02c5368bdf1d01178d19a570"
    assert sha256(source / "checkpoint_epoch_52.pt") == manifest["source_full_checkpoint_sha256"]
    assert set(weights) == set(checkpoint["model_state_dict"])
    assert all(torch.equal(weights[key], checkpoint["model_state_dict"][key]) for key in weights)
    selected = max(history["epochs"], key=lambda row: (
        row["validation"]["accuracy"], -row["validation"]["loss"], -row["epoch"]))
    assert selected["epoch"] == 52
    assert selected["validation"] == manifest["selected_validation"] == test["validation_same_checkpoint"]
    assert test["accuracy"] == .9165 and test["correct"] == 9165 and test["errors"] == 835
    assert test["samples"] == 10000 and test["shirt_recall"] == .756
    assert test["loss"] == 0.2710733623552026

    destination = PROJECT_DIR / "final_model"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("final_model已有内容；请先阅读，不自动覆盖。")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best_cnn.pt", destination / "model.pt")
    assert sha256(destination / "model.pt") == digest
    config = {
        "format_version": 1, "weights_file": "model.pt", "weights_sha256": digest,
        "model": history["model"], "class_names": history["class_names"],
        "preprocessing": {"name": "ToTensor", "input_shape": [1, 28, 28],
                          "dtype": "float32", "pixel_range": [0, 1],
                          "augmentation": None, "normalization": None},
        "inference": {"device": "cpu", "model_mode": "eval", "gradients": False,
                      "output": "10 raw class scores; argmax gives predicted class"},
        "selected_epoch": 52, "selection_rule": history["best_selection_rule"],
        "selected_model_learning_rate_schedule": manifest["selected_model_learning_rate_schedule"],
        "selected_model_parameter_updates": manifest["selected_model_parameter_updates"],
        "training_config": history["config"], "original_training_config": history["original_config"],
        "experiment_completed_epoch": 60,
        "experiment_learning_rate_schedule": history["learning_rate_schedule"],
        "validation_metrics_file": "validation_metrics.json", "test_metrics_file": "test_metrics.json",
    }
    provenance = {
        "source_weight": manifest["source_weight"], "source_weight_sha256": digest,
        "source_full_checkpoint": manifest["source_full_checkpoint"],
        "source_full_checkpoint_sha256": manifest["source_full_checkpoint_sha256"],
        "source_history": str((source / "cnn_training_history.json").relative_to(PROJECT_DIR)),
        "source_history_sha256": sha256(source / "cnn_training_history.json"),
        "source_test_report": "outputs/final_evaluation/test_metrics.json",
        "source_test_report_sha256": sha256(evaluation / "test_metrics.json"),
        "selected_epoch": 52, "selection_rule": history["best_selection_rule"],
        "selected_before_test": manifest["selection_fixed_before_test_inference"],
        "copy_hashes_match": True, "weights_bytes": (destination / "model.pt").stat().st_size,
        "checkpoint_parameters_match": True, "test_re_evaluated_for_packaging": False,
        "historical_checkpoints_in_public_package": False,
        "training_source_sha256": history["source_sha256"],
    }
    for filename, payload in (("config.json", config), ("validation_metrics.json", selected["validation"]),
                              ("test_metrics.json", test), ("provenance.json", provenance)):
        (destination / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                                           + "\n", encoding="utf-8")
    (destination / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in sorted(destination.iterdir()) if path.is_file()),
        encoding="utf-8")
    (destination / "README.md").write_text(
        "# 固定最终模型\n\n"
        "`model.pt`是SimpleCNN第52轮的state_dict，426,764字节（约417KiB），随仓库和ZIP提供。\n\n"
        "第1—45轮学习率0.03，第46—52轮0.01。选择依据是验证准确率最高，"
        "平分取同轮较低验证损失；测试结果未参与选择。完整配置与类别映射见`config.json`，"
        "来源和哈希见`provenance.json`。逐轮完整检查点仅保留在原工作目录，不在公开包中。\n\n"
        "验证准确率91.66%，损失0.260076；既有官方测试准确率91.65%，损失0.271073，"
        "Shirt召回率75.60%。两个指标文件来自同一份权重。此次整理没有重新评估测试集。\n\n"
        "从项目根目录运行：`python src/predict.py --index 0 --count 9`。"
        "首次会下载官方训练数据，演示样本不属于新的测试评估。\n\n"
        "核对文件：`cd final_model && shasum -a 256 -c SHA256SUMS`。\n",
        encoding="utf-8")
    print(f"最终模型：{destination}\n大小：{provenance['weights_bytes']}字节\nSHA-256：{digest}")
    print("记录与第52轮检查点核对通过；未执行数据推理或测试集评估。")


if __name__ == "__main__":
    main()
