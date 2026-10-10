"""读取三个 CNN 学习率实验的已有结果，绘图并选择候选模型；不训练。"""

import hashlib
import json
import math

from analyze_errors import summarize_predictions
from compare_models import read_json, recorded_training_seconds
from data import OUTPUT_DIR, PROJECT_DIR

import matplotlib.pyplot as plt


EXPERIMENTS = (
    (0.003, "cnn_sgd_lr0003_15epochs"),
    (0.01, "cnn_sgd_15epochs"),
    (0.03, "cnn_sgd_lr003_15epochs"),
)
COMPARISON_DIR = OUTPUT_DIR / "experiments" / "cnn_learning_rate_comparison"
CONFIG_KEYS = (
    "model", "seed", "split_seed", "shuffle_seed", "preprocessing", "initialization",
    "optimizer", "loss_function", "batch_size", "num_workers", "device", "cpu_threads",
    "epochs_requested", "train_samples", "validation_samples", "batches_per_epoch",
    "validation_batches", "python", "versions", "class_names", "test_set_used",
)


def inspect_experiment(directory, learning_rate):
    """所有指标对应同一个已保存、已重新加载的最佳轮次。"""
    history = read_json("cnn_training_history.json", directory)
    assert history["learning_rate"] == learning_rate
    status = history.get("status", "completed")  # 原 0.01 历史早于 status 字段。
    if status != "completed":
        return history, None, {
            "learning_rate": learning_rate, "status": status,
            "error": history.get("error"), "completed_epochs": len(history["epochs"]),
            "validation_accuracy": None, "validation_loss": None, "best_epoch": None,
            "shirt_recall": None, "shirt_to_pullover": None,
            "training_seconds": history.get("training_seconds"),
            "eligible_for_selection": False,
        }

    assert len(history["epochs"]) == history["epochs_requested"] == 15
    assert not history["test_set_used"]
    best = max(history["epochs"], key=lambda entry: entry["validation"]["accuracy"])
    assert best["epoch"] == history["best_epoch"]
    assert best["validation"] == history["reload_validation"]
    for entry in history["epochs"]:
        for split in ("train", "validation"):
            values = entry[split]
            assert math.isfinite(values["loss"])
            assert values["accuracy"] == values["correct"] / values["samples"]
        recalled = entry["validation"]["class_recall"]
        assert recalled["label"] == 6 and recalled["total"] == 1033
        assert recalled["recall"] == recalled["correct"] / recalled["total"]

    metrics = read_json("cnn_class_metrics.json", directory)
    predictions = read_json("cnn_validation_predictions.json", directory)
    checkpoint = directory / "best_cnn.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == metrics["checkpoint_sha256"] == predictions["checkpoint_sha256"]
    assert metrics["best_epoch"] == best["epoch"]
    assert metrics["accuracy"] == best["validation"]["accuracy"] == history["best_validation_accuracy"]
    assert metrics["class_names"] == predictions["class_names"] == history["class_names"]
    records = predictions["samples"]
    # 复用现有错误分析的统计，再核对已保存的结果。
    confusion, per_class, _, correct = summarize_predictions(records, metrics["class_names"])
    assert confusion.tolist() == metrics["confusion_matrix"]
    assert per_class == metrics["class_metrics"]
    assert correct == metrics["correct"] == best["validation"]["correct"]
    assert len(records) == history["validation_samples"] == 10000
    shirt = next(item for item in per_class if item["class_name"] == "Shirt")
    recalled = best["validation"]["class_recall"]
    assert (recalled["total"], recalled["correct"], recalled["recall"]) == (
        shirt["total"], shirt["correct"], shirt["recall"],
    )
    pullover_label = metrics["class_names"].index("Pullover")
    result = {
        "learning_rate": learning_rate, "status": "completed", "completed_epochs": 15,
        "validation_accuracy": best["validation"]["accuracy"],
        "validation_loss": best["validation"]["loss"], "best_epoch": best["epoch"],
        "shirt_recall": shirt["recall"], "shirt_correct": shirt["correct"], "shirt_total": shirt["total"],
        "shirt_to_pullover": int(confusion[shirt["label"], pullover_label]),
        "training_seconds": recorded_training_seconds(history),
        "parameter_count": history["model"]["parameter_count"],
        "checkpoint": str(checkpoint.relative_to(PROJECT_DIR)), "checkpoint_sha256": digest,
        "history": str((directory / "cnn_training_history.json").relative_to(PROJECT_DIR)),
        "eligible_for_selection": True,
    }
    return history, records, result


def choose_candidate(results):
    valid = [result for result in results if result["eligible_for_selection"]]
    if not valid:
        raise RuntimeError("没有完成并重新加载验证通过的模型可供选择。")
    # 跨实验准确率相同时优先损失较低者；两者均相同则保留列表中先出现的实验。
    return max(valid, key=lambda result: (result["validation_accuracy"], -result["validation_loss"]))


def main():
    audit = read_json("baseline_audit.json", COMPARISON_DIR)
    assert audit["original_config_matches"] and audit["optimizer_and_metric_calculation_unchanged"]
    histories, records, results = {}, {}, []
    for rate, folder in EXPERIMENTS:
        history, predicted, result = inspect_experiment(OUTPUT_DIR / "experiments" / folder, rate)
        histories[rate], records[rate] = history, predicted
        results.append(result)

    reference = histories[0.01]
    for rate, history in histories.items():
        if records[rate] is None:
            continue
        for key in CONFIG_KEYS:
            assert history[key] == reference[key], f"比较设置不一致：lr={rate}, {key}"
        for index, (left, right) in enumerate(zip(records[0.01], records[rate])):
            assert left["validation_index"] == right["validation_index"] == index
            assert left["official_training_index"] == right["official_training_index"]
            assert left["true_label"] == right["true_label"]
        if rate != 0.01:
            assert history["initial_parameter_sha256"] == audit["reconstructed_initial_parameter_sha256"]
            assert history["initial_loader_state_sha256"] == audit["expected_initial_loader_state_sha256"]
            for actual, expected in zip(history["epochs"], audit["expected_loader_states"]):
                assert actual["epoch"] == expected["epoch"]
                assert actual["loader_state_before_sha256"] == expected["before"]
                assert actual["loader_state_after_sha256"] == expected["after"]
    if records[0.003] is not None and records[0.03] is not None:
        assert histories[0.003]["source_sha256"] == histories[0.03]["source_sha256"]

    selected = choose_candidate(results)
    selected_history = histories[selected["learning_rate"]]
    candidate = {
        "selection_rule": "Highest best validation accuracy; ties prefer lower loss from the same saved epoch.",
        "within_experiment_rule": "Save on strictly higher validation accuracy; ties keep earlier epoch.",
        "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
        "best_epoch": selected["best_epoch"],
        "config": {key: selected_history[key] for key in CONFIG_KEYS},
        "learning_rate": selected["learning_rate"],
        "validation_metrics": selected_history["epochs"][selected["best_epoch"] - 1]["validation"],
        "shirt_to_pullover": selected["shirt_to_pullover"],
        "source_history": selected["history"], "test_set_used": False,
        "scope": "Candidate chosen from these three fixed-seed validation experiments; stop tuning here.",
    }

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for rate, history in histories.items():
        epochs = history["epochs"]
        numbers = [entry["epoch"] for entry in epochs]
        for key, axis, scale, ylabel in (
            ("loss", axes[0], 1, "Validation cross-entropy loss"),
            ("accuracy", axes[1], 100, "Validation accuracy (%)"),
        ):
            values = [entry["validation"][key] * scale for entry in epochs]
            label = f"SGD lr={rate:g}"
            if records[rate] is None:
                label += " (stopped)"
            line, = axis.plot(numbers, values, marker="o", label=label)
            result = next(item for item in results if item["learning_rate"] == rate)
            if result["eligible_for_selection"]:
                best_value = result["validation_loss"] if key == "loss" else result["validation_accuracy"] * 100
                axis.scatter(result["best_epoch"], best_value, marker="*", s=140, color=line.get_color(), zorder=4)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_xticks(range(1, 16))
            axis.grid(alpha=0.3)
            axis.legend(title="Star = saved best-accuracy epoch")
    figure.suptitle("CNN learning-rate comparison: 15 epochs | CPU | seed 42 | batch 64")
    figure.tight_layout()
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = COMPARISON_DIR / "learning_rate_comparison.png"
    figure.savefig(plot_path, dpi=150, facecolor="white", transparent=False)
    plt.close(figure)
    for filename, payload in (
        ("learning_rate_comparison.json", {
            "changed_factor": "SGD learning rate only: 0.003, 0.01, 0.03",
            "matched_config": {key: reference[key] for key in CONFIG_KEYS},
            "models": results, "selected_checkpoint": selected["checkpoint"],
            "baseline_audit": "baseline_audit.json", "limitations": audit["limitations"],
        }),
        ("candidate_model.json", candidate),
    ):
        (COMPARISON_DIR / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("三个学习率各自最佳模型的实际结果（指标来自各自同一轮）：")
    for result in results:
        if not result["eligible_for_selection"]:
            print(f"lr={result['learning_rate']}: {result['status']}；{result['error']}；未列为候选。")
            continue
        seconds = result["training_seconds"]
        time_text = f"{seconds:.2f}s" if seconds is not None else "缺失"
        print(f"lr={result['learning_rate']:g} | 最佳第 {result['best_epoch']} 轮 | "
              f"验证准确率 {result['validation_accuracy']:.2%} | 验证损失 {result['validation_loss']:.6f} | "
              f"Shirt 召回率 {result['shirt_recall']:.2%} | Shirt -> Pullover {result['shirt_to_pullover']} | 耗时 {time_text}")
    print(f"候选模型：{selected['checkpoint']}；学习率 {selected['learning_rate']:g}，第 {selected['best_epoch']} 轮。")
    print(f"已保存：{plot_path}\n{COMPARISON_DIR / 'learning_rate_comparison.json'}\n{COMPARISON_DIR / 'candidate_model.json'}")
    print("未训练、未评估测试集；原 0.01 实验未重跑。历史状态摘要缺失等比较限制保存在报告中。")


if __name__ == "__main__":
    main()
