"""只读取已有结果，对比 MLP、CNN 五轮及可选的 CNN 十五轮实验。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from data import OUTPUT_DIR, PROJECT_DIR


os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(filename, directory=OUTPUT_DIR):
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}，请先完成相应的训练或分析。")
    return json.loads(path.read_text(encoding="utf-8"))


def recorded_training_seconds(history):
    """只用已记录的完整正式训练耗时，缺失时返回 None，不估算。"""
    seconds = history.get("training_seconds")
    if seconds is None:
        return None
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("训练耗时记录无效，请检查历史文件。")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        help="可选的 CNN 十五轮结果目录；相对路径以项目为基准，位于 outputs 内。")
    args = parser.parse_args()
    experiment_dir = None
    if args.experiment_dir is not None:
        experiment_dir = (PROJECT_DIR / args.experiment_dir).resolve()
        if not experiment_dir.is_relative_to(OUTPUT_DIR.resolve()):
            parser.error("实验目录必须位于本项目 outputs 内。")
    directories = {"mlp": OUTPUT_DIR, "cnn": OUTPUT_DIR}
    model_types = {"mlp": "mlp", "cnn": "cnn"}
    labels = {"mlp": "MLP (5 epochs)", "cnn": "CNN (5 epochs)"}
    histories = {
        "mlp": read_json("training_history.json"),
        "cnn": read_json("cnn_training_history.json"),
    }
    if experiment_dir is not None:
        directories["cnn_15"] = experiment_dir
        model_types["cnn_15"] = "cnn"
        labels["cnn_15"] = "CNN (15 epochs)"
        histories["cnn_15"] = read_json("cnn_training_history.json", experiment_dir)
        assert histories["cnn_15"]["epochs_requested"] == 15
        assert histories["cnn_15"]["model"] == histories["cnn"]["model"]
        assert histories["cnn_15"]["shuffle_seed"] == histories["cnn"]["seed"] == 42
        assert histories["cnn_15"]["preprocessing"] == "ToTensor() only"
    metrics = {
        name: read_json(f"{model_types[name]}_class_metrics.json", directories[name])
        for name in histories
    }
    predictions = {
        name: read_json(f"{model_types[name]}_validation_predictions.json", directories[name])
        for name in histories
    }
    # 设置一致只是一次对照的条件，不意味着双方已经各自充分调优。
    setting_keys = (
        "seed", "split_seed", "learning_rate", "optimizer", "loss_function",
        "batch_size", "num_workers", "device",
        "train_samples", "validation_samples", "batches_per_epoch", "validation_batches",
        "cpu_threads", "python",
    )
    assert histories["mlp"]["epochs_requested"] == histories["cnn"]["epochs_requested"] == 5
    for name, history in histories.items():
        for key in setting_keys:
            assert history[key] == histories["mlp"][key], f"设置不同：{name} {key}"
        assert history["versions"] == histories["mlp"]["versions"]
        assert metrics[name]["class_names"] == metrics["mlp"]["class_names"]
        reference = predictions["mlp"]["samples"]
        records = predictions[name]["samples"]
        assert len(records) == len(reference) == history["validation_samples"]
        # 逐张核对顺序、原始索引及真实标签，避免比较不同的验证集。
        for index, (left, right) in enumerate(zip(reference, records)):
            assert left["validation_index"] == right["validation_index"] == index
            assert left["official_training_index"] == right["official_training_index"]
            assert left["true_label"] == right["true_label"]

    summary = {}
    for name, history in histories.items():
        result = metrics[name]
        assert not history["test_set_used"] and not result["test_set_used"]
        assert result["accuracy"] == history["best_validation_accuracy"]
        assert result["accuracy"] == history["reload_validation"]["accuracy"]
        assert [entry["epoch"] for entry in history["epochs"]] == list(range(1, history["epochs_requested"] + 1))
        best = max(history["epochs"], key=lambda entry: entry["validation"]["accuracy"])
        assert best["epoch"] == result["best_epoch"] == history["best_epoch"]
        checkpoint_hash = hashlib.sha256((directories[name] / f"best_{model_types[name]}.pt").read_bytes()).hexdigest()
        assert checkpoint_hash == result["checkpoint_sha256"] == predictions[name]["checkpoint_sha256"]
        assert sum(record["true_label"] == record["predicted_label"] for record in predictions[name]["samples"]) == result["correct"]
        shirt = next(item for item in result["class_metrics"] if item["class_name"] == "Shirt")
        pullover_label = result["class_names"].index("Pullover")
        if "class_recall" in best["validation"]:
            recalled = best["validation"]["class_recall"]
            assert recalled["label"] == shirt["label"]
            assert recalled["total"] == shirt["total"]
            assert recalled["correct"] == shirt["correct"]
            assert recalled["recall"] == shirt["recall"]
        summary[name] = {
            "best_epoch": history["best_epoch"],
            "epochs_requested": history["epochs_requested"],
            "validation_accuracy": result["accuracy"],
            "trainable_parameters": history["model"]["parameter_count"],
            "training_seconds": recorded_training_seconds(history),
            "shirt_recall": shirt["recall"],
            "shirt_correct": shirt["correct"], "shirt_total": shirt["total"],
            "shirt_to_pullover": result["confusion_matrix"][shirt["label"]][pullover_label],
            "checkpoint_sha256": checkpoint_hash,
        }
        if history["epochs_requested"] == 5:
            summary[name]["five_epoch_training_seconds"] = summary[name]["training_seconds"]
    assert len({result["shirt_total"] for result in summary.values()}) == 1
    differences = {
        "validation_accuracy_percentage_points": 100 * (summary["cnn"]["validation_accuracy"] - summary["mlp"]["validation_accuracy"]),
        "shirt_recall_percentage_points": 100 * (summary["cnn"]["shirt_recall"] - summary["mlp"]["shirt_recall"]),
        "trainable_parameters": summary["cnn"]["trainable_parameters"] - summary["mlp"]["trainable_parameters"],
        "shirt_to_pullover": summary["cnn"]["shirt_to_pullover"] - summary["mlp"]["shirt_to_pullover"],
    }
    print("同一验证集上的实际对照（按验证准确率选择各自最佳模型）：")
    for name, result in summary.items():
        seconds = result["training_seconds"]
        time_text = f"{seconds:.2f}s" if seconds is not None else "缺失（不估算，不重新训练）"
        print(
            f"{labels[name]}: 最佳第 {result['best_epoch']} 轮 | 验证准确率 {result['validation_accuracy']:.2%} | "
            f"参数 {result['trainable_parameters']:,} | 正式训练耗时 {time_text} | "
            f"Shirt 召回率 {result['shirt_recall']:.2%} | Shirt -> Pullover {result['shirt_to_pullover']} 张"
        )
    print(f"CNN - MLP：验证准确率 {differences['validation_accuracy_percentage_points']:+.2f} 个百分点，"
          f"Shirt 召回率 {differences['shirt_recall_percentage_points']:+.2f} 个百分点。")
    print("训练耗时均包含每轮训练、验证及最佳参数保存，未使用完整命令耗时代替。")

    extra_report = {}
    if experiment_dir is not None:
        changed = summary["cnn_15"]
        comparisons = {}
        for reference in ("mlp", "cnn"):
            previous = summary[reference]
            delta = {
                "validation_accuracy_percentage_points": 100 * (changed["validation_accuracy"] - previous["validation_accuracy"]),
                "shirt_recall_percentage_points": 100 * (changed["shirt_recall"] - previous["shirt_recall"]),
                "shirt_to_pullover": changed["shirt_to_pullover"] - previous["shirt_to_pullover"],
            }
            comparisons[reference] = delta
            print(f"CNN 15轮 - {labels[reference]}：准确率 {delta['validation_accuracy_percentage_points']:+.2f} 个百分点，"
                  f"Shirt 召回率 {delta['shirt_recall_percentage_points']:+.2f} 个百分点，"
                  f"Shirt -> Pullover {delta['shirt_to_pullover']:+d} 张。")
        # 比较前五轮的实测指标，不比较受机器负载影响的时间。
        prefix_checks = []
        for old, new in zip(histories["cnn"]["epochs"], histories["cnn_15"]["epochs"]):
            matches = all(
                math.isclose(old[split][key], new[split][key], rel_tol=1e-6, abs_tol=1e-8)
                for split in ("train", "validation") for key in ("loss", "accuracy")
            )
            prefix_checks.append({"epoch": old["epoch"], "metrics_match": matches})
        transitions = []
        new_epochs = histories["cnn_15"]["epochs"]
        for before, after in zip(new_epochs, new_epochs[1:]):
            if after["train"]["loss"] < before["train"]["loss"] and after["validation"]["loss"] > before["validation"]["loss"]:
                transitions.append({"from_epoch": before["epoch"], "to_epoch": after["epoch"],
                                    "train_loss_before": before["train"]["loss"], "train_loss_after": after["train"]["loss"],
                                    "validation_loss_before": before["validation"]["loss"], "validation_loss_after": after["validation"]["loss"]})
        extra_report = {"cnn_15_minus": comparisons, "cnn_changed_factor": "epochs_requested: 5 -> 15",
                        "first_five_epoch_checks": prefix_checks,
                        "training_loss_down_validation_loss_up": transitions}
        print(f"前五轮指标复现：{all(item['metrics_match'] for item in prefix_checks)}。")
        print(f"训练损失下降且验证损失上升的相邻轮次：{[(item['from_epoch'], item['to_epoch']) for item in transitions]}。")

    figure, axis = plt.subplots(figsize=(10, 5))
    for name, history in histories.items():
        epochs = [entry["epoch"] for entry in history["epochs"]]
        accuracies = [entry["validation"]["accuracy"] * 100 for entry in history["epochs"]]
        style = {"linestyle": "--", "zorder": 3} if experiment_dir is not None and name == "cnn" else {}
        line, = axis.plot(epochs, accuracies, marker="o", label=labels[name], **style)
        best_epoch = history["best_epoch"]
        axis.scatter(best_epoch, history["best_validation_accuracy"] * 100,
                     marker="*", s=160, color=line.get_color(), zorder=3)
    axis.set_xticks(range(1, max(h["epochs_requested"] for h in histories.values()) + 1))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)")
    title = "MLP vs CNN: one matched-settings run" if experiment_dir is None else "CNN epoch experiment: only epoch count changes"
    axis.set_title(f"{title}\nCPU | seed 42 | SGD lr=0.01 | batch=64")
    axis.grid(alpha=0.3)
    axis.legend(title="Star = best epoch")
    figure.tight_layout()
    output_dir = experiment_dir if experiment_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "model_comparison" if experiment_dir is not None else "mlp_cnn_comparison"
    plot_path = output_dir / f"{stem}.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    report = {
        "matched_settings": {key: histories["mlp"][key] for key in setting_keys},
        "same_validation_records": True,
        "models": summary, "cnn_minus_mlp": differences,
        "timing_scope": "Recorded training_seconds: all formal epochs including validation and best checkpoint saves.",
        "limitation": (
            "One fixed-seed CNN epoch-count experiment (5 vs 15); MLP 5 epochs is a historical reference. "
            "No independent tuning, repeated seeds or test-set evaluation."
            if experiment_dir is not None else
            "One run under the same settings; not independently tuned best performance for either model. No test-set evaluation."
        ),
        **extra_report,
    }
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存：\n{plot_path}\n{json_path}")
    if experiment_dir is not None:
        print("CNN 对照只改变训练轮数；MLP 是此前五轮的参考。一次实验不能推广为模型的普遍优劣。")
    else:
        print("这是相同设置下的一次对照，不代表两个模型各自调优后的最佳能力。")


if __name__ == "__main__":
    main()
