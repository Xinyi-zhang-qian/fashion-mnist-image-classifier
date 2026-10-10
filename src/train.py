"""在 CPU 上复用同一训练流程，训练 MLP 或 CNN 并重新加载验证。"""

import argparse
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch import nn

from data import (
    BATCH_SIZE,
    NUM_WORKERS,
    OUTPUT_DIR,
    PROJECT_DIR,
    SEED,
    create_train_loader,
    create_validation_loader,
    load_training_datasets,
)
from model import SimpleClassifier
from cnn_model import SimpleCNN


LEARNING_RATE = 0.01
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def average_metrics(loss_sum, correct, samples, batches):
    """损失按样本数平均，准确率按正确样本数计算；最后一批也计入。"""
    if samples == 0:
        raise ValueError("没有读取到样本，无法计算指标。")
    metrics = {
        "loss": loss_sum / samples,
        "accuracy": correct / samples,  # 保存 0–1 的比例；打印时转换成百分比。
        "correct": correct,
        "samples": samples,
        "batches": batches,
    }
    if not math.isfinite(metrics["loss"]):
        raise RuntimeError("损失出现非有限值，请先检查训练流程。")
    return metrics


class NumericalError(RuntimeError):
    """发现非有限数值后立即停止，不继续对异常数据更新参数。"""


def check_finite(tensors, phase, batch):
    for tensor in tensors:
        if tensor is not None and not torch.isfinite(tensor).all().item():
            raise NumericalError(f"{phase}：第 {batch} 个批次出现非有限数值。")


def tensor_fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_fingerprint(model):
    """记录初始化参数的摘要；不取样、不改变任何随机状态。"""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_history(path, history):
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """遍历训练批次：清梯度、前向、损失、反向传播、更新参数。"""
    model.train()  # 每轮训练前恢复训练模式。
    loss_sum, correct, samples, batch_count = 0.0, 0, 0, 0
    batches = loader if max_batches is None else islice(loader, max_batches)
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # 原始分数，不加 Softmax。
        check_finite((outputs,), "训练输出", batch_count + 1)
        loss = criterion(outputs, labels)
        check_finite((loss,), "训练损失", batch_count + 1)
        loss.backward()
        check_finite((p.grad for p in model.parameters()), "参数梯度", batch_count + 1)
        optimizer.step()
        check_finite((p.detach() for p in model.parameters()), "更新后参数", batch_count + 1)

        sample_count = labels.size(0)
        # loss 是本批次的平均值，乘实际人数后再累计。
        loss_sum += loss.item() * sample_count
        correct += (outputs.detach().argmax(dim=1) == labels).sum().item()
        samples += sample_count
        batch_count += 1
    return average_metrics(loss_sum, correct, samples, batch_count)


def validate(model, loader, criterion, device, max_batches=None, recall_label=None):
    """验证只计算指标；不记录计算图，不反向传播，不调用优化器。"""
    model.eval()
    loss_sum, correct, samples, batch_count = 0.0, 0, 0, 0
    class_total, class_correct = 0, 0
    batches = loader if max_batches is None else islice(loader, max_batches)
    with torch.no_grad():
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            assert not outputs.requires_grad
            check_finite((outputs,), "验证输出", batch_count + 1)
            loss = criterion(outputs, labels)
            check_finite((loss,), "验证损失", batch_count + 1)
            sample_count = labels.size(0)
            loss_sum += loss.item() * sample_count
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            if recall_label is not None:
                selected = labels == recall_label
                class_total += selected.sum().item()
                class_correct += (selected & (predictions == labels)).sum().item()
            samples += sample_count
            batch_count += 1
    metrics = average_metrics(loss_sum, correct, samples, batch_count)
    if recall_label is not None:
        metrics["class_recall"] = {
            "label": recall_label, "total": class_total, "correct": class_correct,
            "recall": class_correct / class_total if class_total else None,
        }
    return metrics


def check_flow(
    train_dataset, validation_dataset, device, model_class=SimpleClassifier,
    learning_rate=LEARNING_RATE,
):
    """用临时模型检查 3 个训练批次、2 个验证批次，不保存参数。"""
    torch.manual_seed(SEED)
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    before_training = [p.detach().clone() for p in model.parameters()]
    started = perf_counter()
    train_metrics = train_one_epoch(
        model, create_train_loader(train_dataset), criterion, optimizer, device,
        max_batches=3,
    )
    assert model.training
    assert train_metrics["batches"] == 3 and train_metrics["samples"] == 192
    assert any(
        not torch.equal(before, parameter.detach())
        for before, parameter in zip(before_training, model.parameters())
    )
    # 记录训练留下的参数和梯度，确认验证既不更新参数，也不改变梯度。
    before_validation = [p.detach().clone() for p in model.parameters()]
    gradients_before = [p.grad.detach().clone() for p in model.parameters()]
    validation_metrics = validate(
        model, create_validation_loader(validation_dataset), criterion, device,
        max_batches=2,
    )
    assert not model.training
    assert validation_metrics["batches"] == 2 and validation_metrics["samples"] == 128
    for parameter, before, gradient in zip(
        model.parameters(), before_validation, gradients_before
    ):
        assert torch.equal(parameter.detach(), before)
        assert torch.equal(parameter.grad, gradient)
    result = {
        "train": train_metrics,
        "validation": validation_metrics,
        "seconds": perf_counter() - started,
    }
    print(
        f"流程检查通过：3 个训练批次、2 个验证批次；"
        f"训练损失 {train_metrics['loss']:.6f}，"
        f"验证损失 {validation_metrics['loss']:.6f}，"
        f"耗时 {result['seconds']:.2f}s。", flush=True,
    )
    print("已确认：训练更新参数；验证不计算新梯度、不改变参数。", flush=True)
    return result


def save_curves(
    epochs, best_epoch, model_name="MLP", filename="training_curves.png",
    output_dir=OUTPUT_DIR,
):
    """绘制损失与准确率曲线，图中文字使用英文。"""
    numbers = [entry["epoch"] for entry in epochs]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, axis, ylabel in zip(
        ("loss", "accuracy"), axes, ("Mean cross-entropy loss", "Accuracy (%)")
    ):
        scale = 100 if key == "accuracy" else 1
        for split, label in (("train", "Training"), ("validation", "Validation")):
            axis.plot(
                numbers, [entry[split][key] * scale for entry in epochs],
                marker="o", label=label,
            )
        axis.axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_xticks(numbers)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(f"FashionMNIST {model_name}: training and validation (CPU)")
    figure.tight_layout()
    path = output_dir / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_validation_predictions(
    model, dataset, class_names, device, best_epoch,
    model_name="MLP", filename="validation_predictions.png",
    output_dir=OUTPUT_DIR,
):
    """始终展示固定验证划分的前 9 张，不按预测结果挑选样本。"""
    examples = [dataset[index] for index in range(9)]
    images = torch.stack([image for image, _ in examples])
    labels = torch.tensor([label for _, label in examples])
    model.eval()
    with torch.no_grad():
        predictions = model(images.to(device)).argmax(dim=1).cpu()
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    figure.suptitle(
        f"Reloaded best {model_name} (epoch {best_epoch}): fixed validation samples\n"
        "Green = correct; red = wrong. This is not a test-set evaluation.",
        fontsize=13,
    )
    for index, axis in enumerate(axes.flatten()):
        true_label, predicted_label = labels[index].item(), predictions[index].item()
        is_correct = true_label == predicted_label
        axis.imshow(images[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"True: {class_names[true_label]} ({true_label})\n"
            f"Pred: {class_names[predicted_label]} ({predicted_label})\n"
            f"{'Correct' if is_correct else 'Wrong'}",
            color="#187345" if is_correct else "#b72d2d", fontsize=10,
        )
        axis.axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    path = output_dir / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "file": str(path.relative_to(PROJECT_DIR)),
        "official_training_indices": [int(index) for index in dataset.indices[:9]],
        "true_labels": labels.tolist(),
        "predicted_labels": predictions.tolist(),
        "correct": (predictions == labels).sum().item(),
    }


def save_shirt_curve(epochs, best_epoch, output_dir, model_name):
    """召回率来自每轮完整验证，不额外遍历或更新模型。"""
    numbers = [entry["epoch"] for entry in epochs]
    recalls = [entry["validation"]["class_recall"]["recall"] * 100 for entry in epochs]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(numbers, recalls, marker="o", label="Shirt validation recall")
    axis.axvline(best_epoch, color="gray", linestyle="--", label="Best accuracy epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Shirt recall (%)")
    axis.set_ylim(0, 100)
    axis.set_xticks(numbers)
    axis.set_title(f"FashionMNIST {model_name}: Shirt recall on the fixed validation set")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "shirt_recall.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("训练轮数必须至少为 1。")
    return number


def positive_learning_rate(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("学习率必须是大于 0 的有限数值。")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp", help="模型类型，默认 mlp。")
    parser.add_argument("--epochs", type=positive_integer, default=5, help="训练轮数，默认 5。")
    parser.add_argument("--lr", type=positive_learning_rate, default=LEARNING_RATE,
                        help="SGD 学习率，默认 0.01。")
    parser.add_argument("--check-only", action="store_true", help="只做少量批次流程检查，不保存结果。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="结果目录；相对路径以项目为基准，必须位于 outputs 内。")
    args = parser.parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    if not output_dir.is_relative_to(OUTPUT_DIR.resolve()):
        parser.error("结果目录必须位于本项目 outputs 内。")
    model_class = SimpleClassifier if args.model == "mlp" else SimpleCNN
    model_name = args.model.upper()
    history_filename = "training_history.json" if args.model == "mlp" else "cnn_training_history.json"
    curves_filename = "training_curves.png" if args.model == "mlp" else "cnn_training_curves.png"
    predictions_filename = "validation_predictions.png" if args.model == "mlp" else "cnn_validation_predictions.png"
    device = torch.device("cpu")
    print(f"Python 解释器：{sys.executable}\n项目目录：{PROJECT_DIR}", flush=True)
    print(f"CPU；种子 {SEED}；SGD 学习率 {args.lr}；批次大小 {BATCH_SIZE}", flush=True)
    # 复用已验证的固定划分，完全不调用测试集入口。
    train_dataset, validation_dataset = load_training_datasets()
    assert len(train_dataset) == 50_000 and len(validation_dataset) == 10_000
    assert not set(train_dataset.indices).intersection(validation_dataset.indices)
    class_names = train_dataset.dataset.classes
    shirt_label = class_names.index("Shirt")
    assert shirt_label == 6 and class_names.index("Pullover") == 2
    shape_check = None
    if args.model == "cnn":
        from check_cnn import check_shapes
        shape_check = check_shapes(train_dataset, device)
    try:
        flow_check = check_flow(train_dataset, validation_dataset, device, model_class, args.lr)
    except NumericalError as error:
        if not args.check_only:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_history(output_dir / history_filename, {
                "status": "stopped_numerical_error", "stage": "flow_check", "error": str(error),
                "model_name": model_class.__name__, "seed": SEED, "device": str(device),
                "learning_rate": args.lr, "epochs_requested": args.epochs,
                "test_set_used": False, "epochs": [],
            })
        print(f"实验已停止：{error}", flush=True)
        raise SystemExit(2)
    if args.check_only:
        print("仅流程检查结束，未保存模型或正式训练结果。")
        return

    # 丢弃流程检查模型；重新设种子并创建新模型、优化器和 DataLoader。
    torch.manual_seed(SEED)
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    train_loader = create_train_loader(train_dataset)
    validation_loader = create_validation_loader(validation_dataset)
    assert train_loader.generator.initial_seed() == SEED
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"best_{args.model}.pt"
    history_path = output_dir / history_filename
    history = {
        "model": {"name": model_class.__name__, "structure": str(model),
                  "parameter_count": sum(p.numel() for p in model.parameters())},
        "seed": SEED, "device": str(device), "learning_rate": args.lr,
        "optimizer": "SGD (momentum=0, weight_decay=0)",
        "loss_function": "CrossEntropyLoss (raw logits, reduction=mean)",
        "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
        "epochs_requested": args.epochs, "accuracy_unit": "fraction (0 to 1)",
        "train_samples": len(train_dataset), "validation_samples": len(validation_dataset),
        "batches_per_epoch": len(train_loader),
        "validation_batches": len(validation_loader),
        "split_seed": SEED, "test_set_used": False,
        "shuffle_seed": SEED, "preprocessing": "ToTensor() only",
        "initialization": "Fresh model, optimizer and DataLoader; no checkpoint resume",
        "class_names": class_names, "recall_class": "Shirt",
        "cpu_threads": torch.get_num_threads(), "python": sys.version.split()[0],
        "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "matplotlib")},
        "flow_check": flow_check, "epochs": [],
        "status": "running", "initial_parameter_sha256": model_fingerprint(model),
        "initial_loader_state_sha256": tensor_fingerprint(train_loader.generator.get_state()),
        "numerical_checks": "Finite logits, loss, gradients and updated parameters; no gradient clipping",
        "source_sha256": {
            name: hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest()
            for name in ("cnn_model.py", "data.py", "train.py")
        },
    }
    if shape_check is not None:
        history["shape_check"] = shape_check
    print(
        f"\n从头正式训练 {model_name} {args.epochs} 轮：训练 {len(train_dataset):,} 张，"
        f"验证 {len(validation_dataset):,} 张；每轮 {len(train_loader)} 次更新。",
        flush=True,
    )
    best_accuracy, best_epoch = -1.0, None
    history["best_epoch"] = None
    history["checkpoint"] = str(checkpoint_path.relative_to(PROJECT_DIR))
    write_history(history_path, history)
    training_started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} 开始……", flush=True)
        epoch_started = perf_counter()
        loader_state_before = tensor_fingerprint(train_loader.generator.get_state())
        try:
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            validation_metrics = validate(
                model, validation_loader, criterion, device, recall_label=shirt_label,
            )
        except NumericalError as error:
            history.update({
                "status": "stopped_numerical_error", "failed_epoch": epoch,
                "error": str(error), "failed_epoch_seconds": perf_counter() - epoch_started,
                "training_seconds": perf_counter() - training_started,
                "completed_epoch_parameter_updates": sum(e["train"]["batches"] for e in history["epochs"]),
            })
            write_history(history_path, history)
            print(f"实验已停止并记录到 {history_path}：{error}", flush=True)
            raise SystemExit(2)
        assert train_metrics["samples"] == len(train_dataset)
        assert train_metrics["batches"] == len(train_loader)
        assert validation_metrics["samples"] == len(validation_dataset)
        # 仅严格提高时保存；相同准确率保留先出现的一轮。
        is_best = validation_metrics["accuracy"] > best_accuracy
        if is_best:
            torch.save(model.state_dict(), checkpoint_path)
            best_accuracy, best_epoch = validation_metrics["accuracy"], epoch
        seconds = perf_counter() - epoch_started
        history["epochs"].append({
            "epoch": epoch, "train": train_metrics, "validation": validation_metrics,
            "seconds": seconds, "saved_as_best": is_best,
            "loader_state_before_sha256": loader_state_before,
            "loader_state_after_sha256": tensor_fingerprint(train_loader.generator.get_state()),
        })
        history["best_epoch"] = best_epoch
        history["best_validation_accuracy"] = best_accuracy
        write_history(history_path, history)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"训练损失 {train_metrics['loss']:.6f}，准确率 {train_metrics['accuracy']:.2%} | "
            f"验证损失 {validation_metrics['loss']:.6f}，准确率 {validation_metrics['accuracy']:.2%} | "
            f"Shirt 召回率 {validation_metrics['class_recall']['recall']:.2%} | "
            f"耗时 {seconds:.2f}s{' | 保存最佳模型' if is_best else ''}", flush=True,
        )
    history["training_seconds"] = perf_counter() - training_started
    history["total_parameter_updates"] = sum(entry["train"]["batches"] for entry in history["epochs"])
    history["best_epoch"] = best_epoch
    history["best_validation_accuracy"] = best_accuracy
    history["checkpoint"] = str(checkpoint_path.relative_to(PROJECT_DIR))
    history["status"] = "trained"
    write_history(history_path, history)

    # 新建同结构模型，加载的是参数文件，而不是继续使用最后一轮的内存模型。
    reloaded_model = model_class().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded_model.load_state_dict(state_dict)
    reload_started = perf_counter()
    try:
        reloaded_metrics = validate(
            reloaded_model, validation_loader, criterion, device, recall_label=shirt_label,
        )
    except NumericalError as error:
        history.update({"status": "stopped_numerical_error", "stage": "reload_validation", "error": str(error)})
        write_history(history_path, history)
        print(f"重新加载验证已停止并记录：{error}", flush=True)
        raise SystemExit(2)
    best_metrics = history["epochs"][best_epoch - 1]["validation"]
    assert reloaded_metrics["correct"] == best_metrics["correct"]
    assert math.isclose(reloaded_metrics["accuracy"], best_accuracy, abs_tol=1e-12)
    assert math.isclose(reloaded_metrics["loss"], best_metrics["loss"], rel_tol=1e-6, abs_tol=1e-8)
    assert reloaded_metrics["class_recall"] == best_metrics["class_recall"]
    for name, parameter in reloaded_model.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter.detach(), state_dict[name])
    history["reload_validation"] = reloaded_metrics
    history["reload_validation_seconds"] = perf_counter() - reload_started
    print(
        f"\n重新加载核对通过：最佳来自第 {best_epoch} 轮，"
        f"验证损失 {reloaded_metrics['loss']:.6f}，验证准确率 {reloaded_metrics['accuracy']:.2%}。",
        flush=True,
    )
    curves_path = save_curves(history["epochs"], best_epoch, model_name, curves_filename, output_dir)
    shirt_path = save_shirt_curve(history["epochs"], best_epoch, output_dir, model_name)
    history["validation_examples"] = save_validation_predictions(
        reloaded_model, validation_dataset, train_dataset.dataset.classes, device, best_epoch,
        model_name, predictions_filename, output_dir,
    )
    history["shirt_recall_curve"] = str(shirt_path.relative_to(PROJECT_DIR))
    history["status"] = "completed"
    write_history(history_path, history)
    print(
        f"正式训练耗时 {history['training_seconds']:.2f}s，"
        f"共 {history['total_parameter_updates']:,} 次参数更新。", flush=True,
    )
    print(f"已保存：\n{checkpoint_path}\n{history_path}\n{curves_path}\n"
          f"{shirt_path}\n{output_dir / predictions_filename}", flush=True)


if __name__ == "__main__":
    main()
