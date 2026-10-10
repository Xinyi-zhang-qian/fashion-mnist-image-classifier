"""从既有历史/错误分析记录生成实验表，不训练、不推理。"""

import hashlib
import json
from pathlib import Path

from data import OUTPUT_DIR, PROJECT_DIR


EXPERIMENTS = (
    ("MLP · 0.01 · 5轮", "training_history.json", "mlp_class_metrics.json"),
    ("CNN · 0.01 · 5轮", "cnn_training_history.json", "cnn_class_metrics.json"),
    ("CNN · 0.01 · 15轮", "experiments/cnn_sgd_15epochs/cnn_training_history.json", "cnn_class_metrics.json"),
    ("CNN · 0.003 · 15轮", "experiments/cnn_sgd_lr0003_15epochs/cnn_training_history.json", "cnn_class_metrics.json"),
    ("CNN · 0.03 · 15轮（V1）", "experiments/cnn_sgd_lr003_15epochs/cnn_training_history.json", "cnn_class_metrics.json"),
    ("CNN · 0.01 · 45轮", "experiments/v2_sgd_lr001_45epochs/cnn_training_history.json", "cnn_class_metrics.json"),
    ("CNN · 0.03 · 45轮", "experiments/v2_sgd_lr003_45epochs/cnn_training_history.json", "cnn_class_metrics.json"),
    ("CNN · 第45轮后0.03 · 实际54轮", "experiments/v2_sgd_lr003_resume45_max15epochs/cnn_training_history.json", None),
    ("CNN · 第45轮后0.01 · 总60轮（最终）", "experiments/v2_resume45_lr001_15epochs/cnn_training_history.json", None),
)


def main():
    rows = []
    for name, relative, metrics_name in EXPERIMENTS:
        path = OUTPUT_DIR / relative
        history = json.loads(path.read_text(encoding="utf-8"))
        best = history["epochs"][history["best_epoch"] - 1]
        validation = best["validation"]
        assert best["epoch"] == history["best_epoch"]
        assert validation["accuracy"] == history["reload_validation"]["accuracy"]
        assert validation["loss"] == history["reload_validation"]["loss"]
        assert history["test_set_used"] is False
        checkpoint_hash = history.get("checkpoint_sha256")
        hash_source = "training_history"
        shirt = validation.get("class_recall")
        shirt_to_pullover = None
        metrics_path = path.parent / metrics_name if metrics_name else None
        if metrics_path is not None:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert metrics["best_epoch"] == history["best_epoch"]
            assert metrics["accuracy"] == validation["accuracy"] and metrics["correct"] == validation["correct"]
            if checkpoint_hash is not None:
                assert metrics["checkpoint_sha256"] == checkpoint_hash
            else:
                checkpoint_hash = metrics["checkpoint_sha256"]
                hash_source = "class_metrics (early history has no weight hash)"
            local_weight = PROJECT_DIR / metrics["checkpoint"]
            if local_weight.exists():
                assert hashlib.sha256(local_weight.read_bytes()).hexdigest() == checkpoint_hash
            shirt = next(item for item in metrics["class_metrics"] if item["label"] == 6)
            shirt_to_pullover = metrics["confusion_matrix"][6][2]
            if "class_recall" in validation:
                assert shirt["recall"] == validation["class_recall"]["recall"]
        rows.append({
            "experiment": name, "completed_epochs": len(history["epochs"]),
            "best_epoch": best["epoch"], "validation_accuracy": validation["accuracy"],
            "validation_loss": validation["loss"], "shirt_recall": shirt["recall"] if shirt else None,
            "shirt_to_pullover": shirt_to_pullover, "training_seconds": history.get("training_seconds"),
            "parameter_count": history["model"]["parameter_count"],
            "checkpoint_sha256": checkpoint_hash, "checkpoint_hash_source": hash_source,
            "source_history": str(path.relative_to(PROJECT_DIR)),
            "source_class_metrics": str(metrics_path.relative_to(PROJECT_DIR)) if metrics_path else None,
        })
    test = json.loads((PROJECT_DIR / "final_model/test_metrics.json").read_text(encoding="utf-8"))
    assert test["checkpoint_sha256"] == rows[-1]["checkpoint_sha256"]
    assert test["selected_epoch"] == rows[-1]["best_epoch"] == 52
    directory = PROJECT_DIR / "docs"
    directory.mkdir(exist_ok=True)
    payload = {"rows": rows, "final_test": test,
               "note": "每行验证指标均来自本行最佳检查点；仅最终模型有官方测试成绩。"
                       "续训行耗时包含已记录的前45轮，不是选中模型单独耗时。缺失混淆数记null，不重评估。"}
    (directory / "experiment_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# 已有实验结果\n", payload["note"] + "\n",
             "| 实验 | 最佳轮次 | 验证准确率 | 验证损失 | Shirt召回率 | 总实验耗时（秒） |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        recall = f"{row['shirt_recall']:.2%}" if row['shirt_recall'] is not None else "缺失"
        seconds = f"{row['training_seconds']:.2f}" if row['training_seconds'] is not None else "缺失"
        lines.append(f"| {row['experiment']} | {row['best_epoch']} | {row['validation_accuracy']:.2%} | "
                     f"{row['validation_loss']:.6f} | {recall} | {seconds} |")
    lines.extend(["", "同一第52轮模型的官方测试：91.65%，损失0.271073，Shirt召回率75.60%，"
                  "Shirt→Pullover 55张；正确9165、错误835、总数10000。", "",
                  "这是固定划分和随机轨迹下的逐步实验，不代表模型各自充分调优后的上限。"
                  "早期15轮实验缺少完整初始化/样本顺序证明的比较限制仍保留于原比较报告。"])
    text = "\n".join(lines) + "\n"
    (directory / "EXPERIMENT_RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
