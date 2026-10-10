"""冻结已选第52轮CNN，评估官方测试集；不训练、不根据测试成绩选模型。"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import SequentialSampler
from torchvision.datasets import FashionMNIST

from data import DATA_DIR, OUTPUT_DIR, PROJECT_DIR, SEED, create_data_loaders, load_datasets
from cnn_model import SimpleCNN
from analyze_errors import summarize_predictions, plt
from train import average_metrics, model_fingerprint, validate, write_history
from train_v2 import index_hash


SOURCE = OUTPUT_DIR / "experiments/v2_resume45_lr001_15epochs"
SELECTED_EPOCH = 52
SELECTED_SHA256 = "c972ca96c5d63c81998a6d884f57ebc42406f04a02c5368bdf1d01178d19a570"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_selection():
    """在读取任何测试预测之前，核实来源、配置、选择规则与参数。"""
    history_path = SOURCE / "cnn_training_history.json"
    comparison_path = SOURCE / "learning_rate_continuation_comparison.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    weight_path = SOURCE / "best_cnn.pt"
    checkpoint_path = SOURCE / f"checkpoint_epoch_{SELECTED_EPOCH}.pt"
    assert (PROJECT_DIR / history["checkpoint"]).resolve() == weight_path.resolve()
    assert history["best_epoch"] == comparison["selected_epoch"] == SELECTED_EPOCH
    assert sha256(weight_path) == history["checkpoint_sha256"] == SELECTED_SHA256
    assert comparison["selected_checkpoint_sha256"] == SELECTED_SHA256
    best = max(history["epochs"], key=lambda e: (
        e["validation"]["accuracy"], -e["validation"]["loss"], -e["epoch"]))
    assert best["epoch"] == SELECTED_EPOCH
    assert best["validation"] == comparison["selected_validation"] == history["reload_validation"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    assert checkpoint["completed_epoch"] == SELECTED_EPOCH
    assert checkpoint["config"] == history["config"]
    assert set(weights) == set(checkpoint["model_state_dict"])
    assert all(torch.equal(weights[k], checkpoint["model_state_dict"][k]) for k in weights)
    assert all(g["lr"] == 0.01 for g in checkpoint["optimizer_state_dict"]["param_groups"])
    for name, digest in history["source_sha256"].items():
        assert sha256(PROJECT_DIR / "src" / name) == digest, f"训练代码已变化：{name}"
    config = history["config"]
    assert config["device"] == "cpu" and config["preprocessing"] == "ToTensor() only"
    assert config["batch_size"] == 64 and config["num_workers"] == 0
    assert config["seed"] == config["split_seed"] == config["shuffle_seed"] == SEED
    assert sys.version.split()[0] == config["python"]
    for package, expected in config["versions"].items():
        assert version(package) == expected, f"环境版本不一致：{package}"
    return history, checkpoint, weight_path, checkpoint_path


def audit_data_usage(history, training, validation, test):
    """结合已人工阅读的代码和留存实验记录，保存可核查的数据使用证据。"""
    train_indices, val_indices = set(training.indices), set(validation.indices)
    assert training.dataset is validation.dataset and training.dataset.train is True
    assert len(training) == 50_000 and len(validation) == 10_000
    assert len(train_indices) == len(training) and len(val_indices) == len(validation)
    assert not train_indices & val_indices
    assert train_indices | val_indices == set(range(60_000))
    assert index_hash(training.indices) == history["train_indices_sha256"]
    assert index_hash(validation.indices) == history["validation_indices_sha256"]
    assert test.train is False and len(test) == 10_000
    assert str(test.transform) == "ToTensor()"
    assert training.dataset.classes == test.classes == history["class_names"]
    image, _ = test[0]
    assert image.shape == (1, 28, 28) and image.dtype == torch.float32
    assert 0 <= image.min().item() <= image.max().item() <= 1

    # 只审计历史实验/发布/顶层记录；新测试报告不混入“此前是否使用”的证据。
    prior_json = list(OUTPUT_DIR.glob("*.json"))
    for directory in (OUTPUT_DIR / "experiments", OUTPUT_DIR / "releases"):
        prior_json.extend(directory.rglob("*.json"))
    flags = []

    def visit(value, path, key_path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "test_set_used":
                    flags.append({"path": path, "key": key_path + key, "value": item})
                    assert item is False, f"历史记录声明使用了测试集：{path}"
                visit(item, path, key_path + key + ".")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path, key_path + str(index) + ".")

    for path in sorted(prior_json):
        value = json.loads(path.read_text(encoding="utf-8"))
        if path.name.endswith("training_history.json"):
            assert value["test_set_used"] is False
        visit(value, str(path.relative_to(PROJECT_DIR)))

    # 比对压缩包官方MD5，并确认当前被读取的未压缩IDX与这些压缩包完全一致。
    import gzip
    raw_dir = DATA_DIR / "FashionMNIST/raw"
    data_files = []
    for filename, expected_md5 in FashionMNIST.resources:
        archive, raw = raw_dir / filename, raw_dir / Path(filename).stem
        assert hashlib.md5(archive.read_bytes()).hexdigest() == expected_md5
        assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() == sha256(raw)
        data_files.append({"path": str(raw.relative_to(PROJECT_DIR)),
                           "sha256": sha256(raw), "official_archive_md5": expected_md5})
    reviewed = ("data.py", "show_samples.py", "check_data.py", "check_model.py",
                "demo_training_step.py", "train_one_step.py", "train.py", "train_v2.py",
                "continue_cnn.py", "continue_cnn_lower_lr.py", "analyze_errors.py",
                "compare_models.py", "compare_learning_rates.py", "compare_v2.py")
    return {
        "conclusion": "现有代码与留存记录支持：此前官方测试集未用于训练、调参或模型选择。",
        "basis_limit": "这是对现有项目代码和记录的核查，不能证明未留存的外部操作。",
        "early_test_access": "早期准备/前向演示曾加载测试数据、读取数量和类别名称，"
                             "但模型计算和参数更新只使用训练批次，没有迭代测试批次。",
        "validation_origin": "FashionMNIST train=True，seed=42 random_split 50000/10000",
        "train_samples": len(training), "validation_samples": len(validation),
        "overlap_count": 0, "covers_all_60000_training_indices": True,
        "train_indices_sha256": index_hash(training.indices),
        "validation_indices_sha256": index_hash(validation.indices),
        "test_origin": "FashionMNIST train=False；官方独立文件，10000张；download=False",
        "recorded_test_use_flags": flags,
        "historical_json_sha256": {str(p.relative_to(PROJECT_DIR)): sha256(p) for p in sorted(prior_json)},
        "reviewed_source_sha256": {name: sha256(PROJECT_DIR / "src" / name) for name in reviewed},
        "dataset_files": data_files,
    }


def class_statistics(confusion, class_names):
    """真实类是行、预测类是列：TP在对角线，FP来自该列，FN来自该行。"""
    classes = []
    for label, name in enumerate(class_names):
        support = int(confusion[label].sum())
        predicted_count = int(confusion[:, label].sum())
        tp = int(confusion[label, label])
        fp, fn = predicted_count - tp, support - tp
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        classes.append({"label": label, "class_name": name, "support": support,
                        "predicted_count": predicted_count, "correct": tp,
                        "false_positive": fp, "false_negative": fn,
                        "precision": precision, "recall": recall, "f1": f1})
    return classes


def predict_test(model, loader, class_names):
    """保存全部逐图交叉熵，再按真实样本数计算平均；不提前做Softmax。"""
    assert isinstance(loader.sampler, SequentialSampler) and not loader.drop_last
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    records, correct, batches = [], 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            assert images.shape == (len(labels), 1, 28, 28)
            scores = model(images)
            assert scores.shape == (len(labels), 10) and not scores.requires_grad
            losses = criterion(scores, labels)
            assert torch.isfinite(scores).all().item() and torch.isfinite(losses).all().item()
            predicted = scores.argmax(dim=1)
            correct += int((predicted == labels).sum())
            for true, pred, loss in zip(labels.tolist(), predicted.tolist(), losses.tolist()):
                records.append({"test_index": len(records), "true_label": true,
                                "predicted_label": pred, "true_class": class_names[true],
                                "predicted_class": class_names[pred], "is_correct": true == pred,
                                "cross_entropy_loss": loss})
            batches += 1
    metrics = average_metrics(math.fsum(r["cross_entropy_loss"] for r in records),
                              correct, len(records), batches)
    return records, metrics


def save_test_figures(dataset, records, confusion, top_confusions, output_dir):
    names = [f"{i}: {name}" for i, name in enumerate(dataset.classes)]
    figure, axis = plt.subplots(figsize=(12, 10))
    heatmap = axis.imshow(confusion, cmap="Blues", vmin=0)
    axis.set_xticks(range(10), names, rotation=45, ha="right")
    axis.set_yticks(range(10), names)
    axis.set_xlabel("Predicted class (columns)")
    axis.set_ylabel("True class (rows)")
    axis.set_title("Frozen CNN epoch 52 - official test confusion matrix (counts)")
    for true in range(10):
        for pred in range(10):
            count = int(confusion[true, pred])
            axis.text(pred, true, str(count), ha="center", va="center",
                      color="white" if count > confusion.max() / 2 else "black")
    figure.colorbar(heatmap, ax=axis, fraction=.046, pad=.04).set_label("Number of images")
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=150,
                   facecolor="white", transparent=False)
    plt.close(figure)

    # 确定性地覆盖最常见的五种错误方向，每种取官方顺序前3张，再顺序补足16张。
    errors = [r for r in records if not r["is_correct"]]
    examples = []
    for pair in top_confusions:
        examples.extend([r for r in errors if r["true_label"] == pair["true_label"]
                         and r["predicted_label"] == pair["predicted_label"]][:3])
    selected = {r["test_index"] for r in examples}
    examples.extend([r for r in errors if r["test_index"] not in selected][:16 - len(examples)])
    rows = max(1, math.ceil(len(examples) / 4))
    figure, axes = plt.subplots(rows, 4, figsize=(13.6, 3.4 * rows), squeeze=False)
    figure.suptitle("Frozen CNN epoch 52 - examples of common official test errors\n"
                    "First 3 per top-5 confusion direction, then remaining first error. Not frequency-proportional.",
                    fontsize=13)
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, examples):
        image, label = dataset[record["test_index"]]
        assert label == record["true_label"]
        axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"Test {record['test_index']}\nTrue: {record['true_class']} ({label})\n"
                       f"Pred: {record['predicted_class']} ({record['predicted_label']})",
                       color="#b72d2d", fontsize=10)
    if not examples:
        axes[0, 0].text(.5, .5, "No test errors", ha="center", va="center")
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output_dir / "test_errors.png", dpi=150, facecolor="white", transparent=False)
    plt.close(figure)
    return {"rule": "五个最多的有向混淆各取官方测试顺序前3张，再按顺序补足16张；不按外观挑选。",
            "limit": "用于展示主要错误类型；这些图片不按错误频率抽样，不能代替完整统计。",
            "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "final_evaluation")
    args = parser.parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    if output_dir.parent != OUTPUT_DIR.resolve():
        parser.error("评估目录必须是本项目outputs下的独立目录。")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error("结果目录已有文件，请指定新的空目录；不会覆盖。")
    started = perf_counter()
    history, checkpoint, weight_path, checkpoint_path = inspect_selection()
    torch.set_num_threads(history["cpu_threads"])
    torch.manual_seed(SEED)
    training, validation, test = load_datasets()
    audit = audit_data_usage(history, training, validation, test)
    _, val_loader, test_loader = create_data_loaders(training, validation, test)
    model = SimpleCNN().to("cpu")
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    assert str(model) == history["model"]["structure"]
    assert sum(p.numel() for p in model.parameters()) == history["model"]["parameter_count"]
    fingerprint_before = model_fingerprint(model)

    # 在第一次测试推理前复制权重、历史与配置；测试结果不改变选择。
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_weight = output_dir / "best_cnn.pt"
    shutil.copy2(weight_path, frozen_weight)
    assert sha256(frozen_weight) == SELECTED_SHA256
    shutil.copy2(SOURCE / "cnn_training_history.json", output_dir / "source_training_history.json")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_fixed_before_test_inference": True, "selected_epoch": SELECTED_EPOCH,
        "source_weight": str(weight_path.relative_to(PROJECT_DIR)), "weight_sha256": SELECTED_SHA256,
        "frozen_weight": str(frozen_weight.relative_to(PROJECT_DIR)),
        "source_full_checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "source_full_checkpoint_sha256": sha256(checkpoint_path),
        "full_checkpoint_fields": list(checkpoint),
        "experiment_config": history["config"], "original_config": history["original_config"],
        "experiment_learning_rate_schedule": history["learning_rate_schedule"],
        "selected_model_learning_rate_schedule": [{"epochs": [1, 45], "learning_rate": .03},
                                                   {"epochs": [46, 52], "learning_rate": .01}],
        "selected_model_parameter_updates": SELECTED_EPOCH * history["batches_per_epoch"],
        "experiment_completed_epoch": history["epochs"][-1]["epoch"],
        "selected_validation": history["reload_validation"],
        "selection_rule": history["best_selection_rule"],
        "training_source_sha256": history["source_sha256"],
        "evaluation": {"python": sys.version.split()[0], "interpreter": sys.executable,
                       "device": "cpu", "cpu_threads": torch.get_num_threads(),
                       "batch_size": test_loader.batch_size, "num_workers": test_loader.num_workers,
                       "shuffle": False, "drop_last": False, "mode": "eval + torch.inference_mode",
                       "preprocessing": "ToTensor() only", "train": False, "download": False,
                       "criterion": "CrossEntropyLoss on raw logits; mean over all samples",
                       "versions": history["versions"], "model_selection_uses_test": False},
    }
    source_copy = output_dir / "source"
    source_copy.mkdir()
    for name in (*history["source_sha256"], "analyze_errors.py", "evaluate_final.py"):
        shutil.copy2(PROJECT_DIR / "src" / name, source_copy / name)
    manifest["evaluation_source_sha256"] = {p.name: sha256(p) for p in sorted(source_copy.iterdir())}
    write_history(output_dir / "model_manifest.json", manifest)
    write_history(output_dir / "data_audit.json", audit)
    print(f"已冻结第52轮；SHA-256：{SELECTED_SHA256}", flush=True)
    print("历史数据使用核查通过；开始完整官方测试集推理，不训练、不选择模型。", flush=True)

    # 重新核对验证指标，仅检查是否加载了指定检查点；不重新挑选模型。
    reloaded_validation = validate(model, val_loader, nn.CrossEntropyLoss(),
                                   torch.device("cpu"), recall_label=6)
    assert reloaded_validation == history["reload_validation"]
    records, metrics = predict_test(model, test_loader, test.classes)
    confusion, _, top_confusions, correct = summarize_predictions(records, test.classes)
    classes = class_statistics(confusion, test.classes)
    assert metrics["samples"] == len(test) == len(records) == 10_000
    assert metrics["correct"] == correct == int(np.trace(confusion))
    assert int(confusion.sum()) == sum(c["support"] for c in classes) == 10_000
    assert sum(c["predicted_count"] for c in classes) == 10_000
    assert sum(c["correct"] for c in classes) == correct
    assert sum(c["false_positive"] for c in classes) == sum(c["false_negative"] for c in classes) == 10_000 - correct
    assert [r["test_index"] for r in records] == list(range(10_000))
    assert [r["true_label"] for r in records] == test.targets.tolist()
    assert [c["support"] for c in classes] == [1000] * 10
    assert metrics["batches"] == len(test_loader) == 157
    assert model_fingerprint(model) == fingerprint_before
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert sha256(weight_path) == sha256(frozen_weight) == SELECTED_SHA256
    samples = metrics["samples"]
    summary = {
        "dataset": "FashionMNIST official test (train=False)", "selected_epoch": SELECTED_EPOCH,
        "checkpoint_sha256": SELECTED_SHA256, **metrics, "errors": samples - correct,
        "class_names": test.classes, "per_class": classes,
        "confusion_matrix": confusion.tolist(), "matrix_axes": {"rows": "true", "columns": "predicted"},
        "shirt_recall": classes[6]["recall"], "shirt_to_pullover": int(confusion[6, 2]),
        "top_five_confusions": top_confusions,
        "macro_average": {key: math.fsum(c[key] for c in classes) / 10 for key in ("precision", "recall", "f1")},
        "weighted_average": {key: math.fsum(c[key] * c["support"] for c in classes) / samples
                             for key in ("precision", "recall", "f1")},
        "zero_denominator_policy": "precision/recall/F1分母为0时记0；本次10类均有样本和预测。",
        "validation_same_checkpoint": reloaded_validation,
        "test_minus_validation_accuracy_percentage_points": 100 * (metrics["accuracy"] - reloaded_validation["accuracy"]),
        "parameter_updates_during_evaluation": 0, "test_used_for_model_selection": False,
        "limitations": ["单一固定随机轨迹，未做多种子重复实验。",
                        "验证集经过多次调参选择，验证成绩与测试成绩应分开报告。",
                        "仅评估FashionMNIST的28×28灰度小图，不能证明对真实服装照片有效。",
                        "错误图片用于说明主要错误方向，不能代表其出现频率。",
                        "网页与二维分拣暂缓；本次测试只用于最终报告，不据此继续调参。"],
    }
    write_history(output_dir / "test_predictions.json", {
        "dataset": summary["dataset"], "index_definition": "官方测试集索引0—9999，与验证索引属于不同数据集。",
        "checkpoint_sha256": SELECTED_SHA256, "records": records})
    examples = save_test_figures(test, records, confusion, top_confusions, output_dir)
    write_history(output_dir / "error_examples.json", examples)
    np.savetxt(output_dir / "test_confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    summary["evaluation_seconds"] = perf_counter() - started
    write_history(output_dir / "test_metrics.json", summary)
    write_history(output_dir / "evaluation_checks.json", {
        "samples": samples, "correct": correct, "errors": samples - correct,
        "confusion_sum": int(confusion.sum()), "diagonal_sum": int(np.trace(confusion)),
        "class_support_sum": sum(c["support"] for c in classes),
        "class_prediction_sum": sum(c["predicted_count"] for c in classes),
        "all_test_indices_unique_and_complete": True, "official_labels_match": True,
        "validation_matches_saved_epoch52": True, "model_eval": not model.training,
        "parameter_fingerprint_before": fingerprint_before,
        "parameter_fingerprint_after": model_fingerprint(model), "all_parameter_gradients_none": True,
        "source_and_frozen_weight_sha256": SELECTED_SHA256,
    })
    print(f"测试：{samples}张；正确{correct}，错误{samples-correct}；"
          f"准确率{metrics['accuracy']:.2%}；平均交叉熵{metrics['loss']:.9f}")
    print("Class          Support Correct Precision   Recall       F1")
    for c in classes:
        print(f"{c['class_name']:<14} {c['support']:7d} {c['correct']:7d} "
              f"{c['precision']:9.2%} {c['recall']:9.2%} {c['f1']:9.2%}")
    print(f"Shirt召回率：{classes[6]['recall']:.2%}；Shirt -> Pullover：{confusion[6, 2]}张")
    for pair in top_confusions:
        print(f"真实{pair['true_class']} -> 预测{pair['predicted_class']}：{pair['count']}张")
    print(f"所有统计与无参数更新检查通过；结果：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
