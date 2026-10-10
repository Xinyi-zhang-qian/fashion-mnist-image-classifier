"""读取两组 V2 45 轮实验与验证错误分析，区分指定轮指标和截至该轮的最佳指标。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from analyze_errors import summarize_predictions
from compare_models import read_json
from data import OUTPUT_DIR, PROJECT_DIR
from train import write_history

import matplotlib.pyplot as plt


def best_until(epochs, limit):
    eligible = [e for e in epochs if e["epoch"] <= limit]
    return max(eligible, key=lambda e: (e["validation"]["accuracy"], -e["validation"]["loss"], -e["epoch"]))


def window_summary(epochs, start, end):
    selected = [e for e in epochs if start <= e["epoch"] <= end]
    assert len(selected) == end - start + 1
    return {"first_epoch": start, "last_epoch": end,
            "mean_training_loss": fmean(e["train"]["loss"] for e in selected),
            "mean_validation_loss": fmean(e["validation"]["loss"] for e in selected),
            "mean_validation_accuracy": fmean(e["validation"]["accuracy"] for e in selected),
            "mean_shirt_recall": fmean(e["validation"]["class_recall"]["recall"] for e in selected)}


def tail_summary(epochs):
    last = epochs[-10:]
    result = {"epochs": [e["epoch"] for e in last],
              "first_half": window_summary(epochs, 36, 40), "second_half": window_summary(epochs, 41, 45)}
    for split, key in (("train", "loss"), ("validation", "loss"), ("validation", "accuracy")):
        values = [e[split][key] for e in last]
        result[f"{split}_{key}"] = {"first": values[0], "last": values[-1],
            "minimum": min(values), "maximum": max(values), "range": max(values) - min(values),
            "linear_slope_per_epoch": float(np.polyfit(range(36, 46), values, 1)[0]),
            "increases": sum(b > a for a, b in zip(values, values[1:])),
            "decreases": sum(b < a for a, b in zip(values, values[1:]))}
    result["interpretation_limit"] = "Descriptive trends only; no proof of convergence, local optima or future performance."
    return result


def inspect(directory, rate):
    history = read_json("cnn_training_history.json", directory)
    assert history["learning_rate"] == rate and not history["test_set_used"]
    if history["status"] != "completed":
        return history, None, {"learning_rate": rate, "status": history["status"],
                               "completed_epochs": len(history["epochs"]), "error": history.get("error")}
    assert len(history["epochs"]) == history["epochs_requested"] == 45
    epochs = history["epochs"]
    best = best_until(epochs, 45)
    assert best["epoch"] == history["best_epoch"] and best["validation"] == history["reload_validation"]
    metrics = read_json("cnn_class_metrics.json", directory)
    predictions = read_json("cnn_validation_predictions.json", directory)
    weight = directory / "best_cnn.pt"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest == history["checkpoint_sha256"] == metrics["checkpoint_sha256"] == predictions["checkpoint_sha256"]
    matrix, classes, confusions, correct = summarize_predictions(predictions["samples"], history["class_names"])
    assert matrix.tolist() == metrics["confusion_matrix"] and correct == best["validation"]["correct"]
    assert metrics["best_epoch"] == best["epoch"] and metrics["accuracy"] == best["validation"]["accuracy"]
    shirt = classes[6]
    assert shirt["recall"] == best["validation"]["class_recall"]["recall"]
    horizons, milestones = {}, {}
    for limit in (15, 30, 45):
        chosen = best_until(epochs, limit)
        horizons[str(limit)] = {"best_epoch": chosen["epoch"], "validation": chosen["validation"]}
        entry = epochs[limit - 1]
        milestones[str(limit)] = {"epoch": limit, "train": entry["train"],
                                  "validation": entry["validation"], "seconds": entry["seconds"]}
    result = {"learning_rate": rate, "status": history["status"], "best_epoch": best["epoch"],
        "best_validation": best["validation"], "shirt_recall": shirt["recall"],
        "shirt_to_pullover": int(matrix[6, 2]), "top_confusions": confusions,
        "parameter_count": history["model"]["parameter_count"],
        "training_seconds": history["training_seconds"], "checkpoint": str(weight.relative_to(PROJECT_DIR)),
        "checkpoint_sha256": digest, "best_within_horizons": horizons, "milestone_metrics": milestones,
        "last_10_epochs": tail_summary(epochs),
        "late_windows": {"21-30": window_summary(epochs, 21, 30),
                         "31-40": window_summary(epochs, 31, 40),
                         "36-45": window_summary(epochs, 36, 45)}}
    return history, predictions["samples"], result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr001-dir", type=Path, default=Path("outputs/experiments/v2_sgd_lr001_45epochs"))
    parser.add_argument("--lr003-dir", type=Path, default=Path("outputs/experiments/v2_sgd_lr003_45epochs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/v2_comparison"))
    args = parser.parse_args()
    directories = [(PROJECT_DIR / args.lr001_dir).resolve(), (PROJECT_DIR / args.lr003_dir).resolve()]
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    assert all(p.is_relative_to((OUTPUT_DIR / "experiments").resolve()) for p in directories + [output_dir])
    histories, predictions, results = [], [], []
    for directory, rate in zip(directories, (0.01, 0.03)):
        history, records, result = inspect(directory, rate)
        histories.append(history)
        predictions.append(records)
        results.append(result)
    left, right = histories
    config_left, config_right = copy.deepcopy(left["config"]), copy.deepcopy(right["config"])
    for config in (config_left, config_right):
        config.pop("learning_rate")
        config["optimizer_defaults"].pop("lr")
    assert config_left == config_right
    assert left["initial_parameter_sha256"] == right["initial_parameter_sha256"]
    assert left["initial_loader_state_sha256"] == right["initial_loader_state_sha256"]
    common = min(len(left["epochs"]), len(right["epochs"]))
    for a, b in zip(left["epochs"], right["epochs"]):
        assert a["epoch"] == b["epoch"] and a["sample_order"] == b["sample_order"]
        assert a["loader_state_before_sha256"] == b["loader_state_before_sha256"]
        assert a["loader_state_after_sha256"] == b["loader_state_after_sha256"]
    assert left["source_sha256"] == right["source_sha256"]
    if all(records is not None for records in predictions):
        assert [(r["official_training_index"], r["true_label"]) for r in predictions[0]] == [
            (r["official_training_index"], r["true_label"]) for r in predictions[1]]
    release_dir = OUTPUT_DIR / "releases/v1"
    v1 = read_json("release_manifest.json", release_dir)
    assert hashlib.sha256((release_dir / "best_cnn.pt").read_bytes()).hexdigest() == v1["checkpoint_sha256"]
    payload = {"scope": "Two fresh 45-epoch runs; learning rate comparison at equal training budgets.",
        "test_set_used": False, "v1_replaced": False, "v1_validation": v1["validation_metrics"],
        "initial_parameter_sha256": left["initial_parameter_sha256"],
        "sample_order_matching_epochs": common, "common_config": config_left, "models": results,
        "limits": ["One seed and one validation split; validation reused for selection.",
                   "No continuation experiment from one shared epoch-15 state.",
                   "45 epochs ending is not evidence of convergence; curve plateaus do not prove local optima.",
                   "Training loss averages changing parameters within each epoch; validation uses end-of-epoch parameters."]}
    if all(result["status"] == "completed" for result in results):
        payload["best_accuracy_gap_percentage_points"] = {
            str(limit): 100 * (results[1]["best_within_horizons"][str(limit)]["validation"]["accuracy"] -
                              results[0]["best_within_horizons"][str(limit)]["validation"]["accuracy"])
            for limit in (15, 30, 45)}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for history, rate in zip(histories, (0.01, 0.03)):
        epochs = history["epochs"]
        numbers = [e["epoch"] for e in epochs]
        for axis, values, title, ylabel in (
            (axes[0, 0], [e["train"]["loss"] for e in epochs], "Training loss", "Mean cross-entropy loss"),
            (axes[0, 1], [e["validation"]["loss"] for e in epochs], "Validation loss", "Mean cross-entropy loss"),
            (axes[1, 0], [100 * e["validation"]["accuracy"] for e in epochs], "Validation accuracy", "Accuracy (%)"),
            (axes[1, 1], [100 * e["validation"]["class_recall"]["recall"] for e in epochs], "Shirt validation recall", "Recall (%)")):
            axis.plot(numbers, values, marker=".", label=f"SGD lr={rate:g}")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
    for axis in axes.flat:
        axis.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        axis.axvspan(35.5, 45.5, color="gray", alpha=.08, label="Last 10 epochs")
        axis.grid(alpha=.3)
        axis.legend()
    figure.suptitle("V2: independent fresh CNN runs | 45 epochs | CPU | seed 42 | batch 64")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "v2_comparison.png", dpi=150, facecolor="white", transparent=False)
    plt.close(figure)
    write_history(output_dir / "v2_comparison.json", payload)
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("初始参数及实际样本顺序逐轮一致；V1保留，未替换，未使用测试集。", flush=True)


if __name__ == "__main__":
    main()
