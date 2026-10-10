"""加载已训练的 MLP 或 CNN，只分析原验证集的分类错误。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from data import (
    OUTPUT_DIR,
    PROJECT_DIR,
    SEED,
    create_validation_loader,
    load_training_datasets,
)
from model import SimpleClassifier
from cnn_model import SimpleCNN


os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def predict_validation(model, loader, dataset, device):
    """按验证集顺序预测，区分验证集序号与官方训练集原始索引。"""
    assert isinstance(loader.sampler, SequentialSampler)
    assert not loader.drop_last
    records = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            assert not outputs.requires_grad
            assert outputs.shape == (len(labels), 10)
            assert torch.isfinite(outputs).all().item()
            predictions = outputs.argmax(dim=1).cpu().tolist()
            for true_label, predicted_label in zip(labels.tolist(), predictions):
                validation_index = len(records)
                records.append({
                    "validation_index": validation_index,
                    "official_training_index": int(dataset.indices[validation_index]),
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                })
    assert len(records) == len(dataset)
    return records


def summarize_predictions(records, class_names):
    """计数矩阵：真实类别是行，预测类别是列，两个方向分别统计。"""
    class_count = len(class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for record in records:
        true_label, predicted_label = record["true_label"], record["predicted_label"]
        assert 0 <= true_label < class_count and 0 <= predicted_label < class_count
        confusion[true_label, predicted_label] += 1

    class_metrics = []
    for label, class_name in enumerate(class_names):
        total = int(confusion[label].sum())  # 行总和：实际属于这一类的数量。
        correct = int(confusion[label, label])
        class_metrics.append({
            "label": label,
            "class_name": class_name,
            "total": total,
            "correct": correct,
            "recall": correct / total if total else None,
        })

    # 排除对角线；A -> B 和 B -> A 是两个不同条目。并列时按类别编号排序。
    confusions = []
    for true_label in range(class_count):
        for predicted_label in range(class_count):
            count = int(confusion[true_label, predicted_label])
            if true_label != predicted_label and count > 0:
                confusions.append({
                    "true_label": true_label,
                    "true_class": class_names[true_label],
                    "predicted_label": predicted_label,
                    "predicted_class": class_names[predicted_label],
                    "count": count,
                })
    confusions.sort(key=lambda item: (-item["count"], item["true_label"], item["predicted_label"]))
    correct = sum(record["true_label"] == record["predicted_label"] for record in records)
    assert int(confusion.sum()) == len(records)
    assert int(np.trace(confusion)) == correct
    assert sum(item["total"] for item in class_metrics) == len(records)
    return confusion, class_metrics, confusions[:5], correct


def save_confusion_matrix(
    confusion, class_names, model_name="MLP", filename="mlp_confusion_matrix.png",
    output_dir=OUTPUT_DIR,
):
    """绘制全部 100 个计数格子，中文轴名称使用已有 Mac 字体。"""
    figure, axis = plt.subplots(figsize=(12, 10))
    heatmap = axis.imshow(confusion, cmap="Blues", vmin=0)
    ticks = np.arange(len(class_names))
    names = [f"{label}: {name}" for label, name in enumerate(class_names)]
    axis.set_xticks(ticks, labels=names, rotation=45, ha="right", fontsize=10)
    axis.set_yticks(ticks, labels=names, fontsize=11)
    chinese_font = FontProperties(family="Arial Unicode MS", size=13)
    axis.set_xlabel("预测类别 (Predicted class)", fontproperties=chinese_font)
    axis.set_ylabel("真实类别 (True class)", fontproperties=chinese_font)
    axis.set_title(f"{model_name} validation confusion matrix - image counts", fontsize=15, pad=15)
    threshold = confusion.max() / 2
    for true_label in range(len(class_names)):
        for predicted_label in range(len(class_names)):
            count = int(confusion[true_label, predicted_label])
            axis.text(
                predicted_label, true_label, str(count), ha="center", va="center",
                color="white" if count > threshold else "black", fontsize=10,
            )
    colorbar = figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Number of images")
    figure.tight_layout()
    path = output_dir / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_error_examples(
    dataset, error_records, class_names, model_name="MLP", filename="mlp_errors.png",
    output_dir=OUTPUT_DIR,
):
    """仅展示固定顺序的前 16 个错误；标签沿用完整预测的记录。"""
    examples = error_records[:16]
    columns = min(4, len(examples)) if examples else 1
    rows = math.ceil(len(examples) / columns) if examples else 1
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.4 * rows), squeeze=False)
    figure.suptitle(
        f"{model_name}: first {len(examples)} errors in fixed validation order\n"
        "True = actual class; Pred = model prediction. No sample selection by appearance.",
        fontsize=14,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, examples):
        validation_index = record["validation_index"]
        image, label = dataset[validation_index]
        assert label == record["true_label"]
        assert int(dataset.indices[validation_index]) == record["official_training_index"]
        axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(
            f"Val {validation_index} | source {record['official_training_index']}\n"
            f"True: {class_names[label]} ({label})\n"
            f"Pred: {class_names[record['predicted_label']]} ({record['predicted_label']})",
            color="#b72d2d", fontsize=10,
        )
    if not examples:
        axes[0, 0].text(0.5, 0.5, "No validation errors", ha="center", va="center")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_dir / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp", help="模型类型，默认 mlp。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="参数、历史及分析结果所在目录；相对路径以项目为基准，位于 outputs 内。")
    args = parser.parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    if not output_dir.is_relative_to(OUTPUT_DIR.resolve()):
        parser.error("结果目录必须位于本项目 outputs 内。")
    model_class = SimpleClassifier if args.model == "mlp" else SimpleCNN
    model_name = args.model.upper()
    started = perf_counter()
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    checkpoint_path = output_dir / f"best_{args.model}.pt"
    history_path = output_dir / ("training_history.json" if args.model == "mlp" else "cnn_training_history.json")
    if not checkpoint_path.is_file() or not history_path.is_file():
        raise FileNotFoundError("缺少最佳模型参数或训练记录，请检查已有训练产物；本脚本不会训练。")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["split_seed"] == SEED
    expected = history["epochs"][history["best_epoch"] - 1]["validation"]
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    print(f"Python 解释器：{sys.executable}\n项目目录：{PROJECT_DIR}", flush=True)
    print(f"CPU；固定划分种子 {SEED}；加载 {model_name} 第 {history['best_epoch']} 轮的最佳参数。", flush=True)

    # 既有入口只读取官方训练数据，原划分中取出验证部分。
    _, validation_dataset = load_training_datasets()
    validation_loader = create_validation_loader(validation_dataset)
    class_names = validation_dataset.dataset.classes
    assert len(class_names) == 10
    model = model_class().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    parameters_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    records = predict_validation(model, validation_loader, validation_dataset, device)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), parameters_before[name])
        assert parameter.grad is None
    assert not model.training

    confusion, class_metrics, top_confusions, correct = summarize_predictions(records, class_names)
    total = len(records)
    accuracy = correct / total
    assert total == expected["samples"] == history["validation_samples"]
    assert correct == expected["correct"]
    assert math.isclose(accuracy, history["best_validation_accuracy"], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(accuracy, history["reload_validation"]["accuracy"], rel_tol=0, abs_tol=1e-12)
    errors = [record for record in records if record["true_label"] != record["predicted_label"]]
    lowest_three = sorted(
        (item for item in class_metrics if item["recall"] is not None),
        key=lambda item: (item["recall"], item["label"]),
    )[:3]

    print(f"\n完整验证集：{total:,} 张；正确 {correct:,}，错误 {len(errors):,}；准确率 {accuracy:.2%}。")
    print("与此前最佳模型记录一致；预测过程参数不变，未计算参数梯度。")
    print("\n每类统计（召回率 = 该类分对数量 / 该类实际数量）：")
    for item in class_metrics:
        recall_text = f"{item['recall']:.2%}" if item["recall"] is not None else "无该类样本"
        print(f"  {item['label']}: {item['class_name']:<12} 总数 {item['total']:4d}，正确 {item['correct']:4d}，召回率 {recall_text}")
    print("召回率最低的三个类别：")
    for item in lowest_three:
        print(f"  {item['class_name']}: {item['recall']:.2%} ({item['correct']}/{item['total']})")
    print("\n数量最多的五种混淆（两个方向分开计数）：")
    for item in top_confusions:
        print(f"  真实 {item['true_class']} -> 预测 {item['predicted_class']}，共 {item['count']} 张")
    print(f"\n一致性检查：矩阵总和 {confusion.sum()}，对角线总和 {np.trace(confusion)}，各类别总数之和 {sum(item['total'] for item in class_metrics)}。")

    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = save_confusion_matrix(confusion, class_names, model_name, f"{args.model}_confusion_matrix.png", output_dir)
    errors_path = save_error_examples(validation_dataset, errors, class_names, model_name, f"{args.model}_errors.png", output_dir)
    predictions_path = output_dir / f"{args.model}_validation_predictions.json"
    predictions_payload = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "checkpoint_sha256": checkpoint_hash, "split_seed": SEED,
        "class_names": class_names,
        "index_explanation": {
            "validation_index": "固定验证集内的序号，0 到 9999",
            "official_training_index": "对应官方 60000 张训练图片中的原始索引",
        },
        "samples": records,
    }
    predictions_path.write_text(json.dumps(predictions_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path = output_dir / f"{args.model}_class_metrics.json"
    metrics_payload = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "checkpoint_sha256": checkpoint_hash, "best_epoch": history["best_epoch"],
        "split_seed": SEED, "device": str(device), "test_set_used": False,
        "validation_samples": total, "correct": correct, "errors": len(errors),
        "accuracy": accuracy, "expected_accuracy": history["best_validation_accuracy"],
        "rate_unit": "fraction (0 to 1)",
        "class_names": class_names,
        "confusion_matrix_axes": {"rows": "真实类别", "columns": "预测类别"},
        "confusion_matrix": confusion.tolist(), "class_metrics": class_metrics,
        "lowest_recall_classes": lowest_three, "top_confusions": top_confusions,
        "shown_errors": errors[:16],
        "predictions_file": str(predictions_path.relative_to(PROJECT_DIR)),
        "analysis_seconds": perf_counter() - started,
        "checks": {"matches_training_history": True, "parameters_unchanged": True,
                   "gradients_not_computed": True, "matrix_total_matches": True,
                   "diagonal_matches_correct": True, "class_totals_match": True},
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"前 {min(16, len(errors))} 个错误的验证集序号：{[record['validation_index'] for record in errors[:16]]}")
    print(f"已保存：\n{confusion_path}\n{metrics_path}\n{errors_path}\n{predictions_path}")
    print(f"分析耗时约 {metrics_payload['analysis_seconds']:.2f}s；只做预测与统计，没有训练。")


if __name__ == "__main__":
    main()
