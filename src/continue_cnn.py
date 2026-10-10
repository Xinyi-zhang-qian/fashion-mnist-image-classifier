"""从 V2 第45轮完整状态续训最多15轮，单独保存，并按双指标耐心8轮早停。"""

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn

from cnn_model import SimpleCNN
from data import (BATCH_SIZE, NUM_WORKERS, OUTPUT_DIR, PROJECT_DIR,
                  create_train_loader, create_validation_loader, load_training_datasets)
from train import (NumericalError, model_fingerprint, tensor_fingerprint,
                   train_one_epoch, validate, write_history)
from train_v2 import (RecordingDataset, index_hash, is_better, load_checkpoint,
                      make_checkpoint, restore_random_states)

import matplotlib.pyplot as plt

SOURCE_DIR = OUTPUT_DIR / "experiments/v2_sgd_lr003_45epochs"
DEFAULT_OUTPUT = OUTPUT_DIR / "experiments/v2_sgd_lr003_resume45_max15epochs"
PATIENCE = 8


def update_patience(state, metrics):
    """分别追踪两个最好值。平分不算改善，两项都等待至少8轮时早停。"""
    loss_improved = metrics["loss"] < state["best_loss"]
    accuracy_improved = metrics["accuracy"] > state["best_accuracy"]
    state["loss_wait"] = 0 if loss_improved else state["loss_wait"] + 1
    state["accuracy_wait"] = 0 if accuracy_improved else state["accuracy_wait"] + 1
    state["best_loss"] = min(state["best_loss"], metrics["loss"])
    state["best_accuracy"] = max(state["best_accuracy"], metrics["accuracy"])
    return (loss_improved, accuracy_improved,
            state["loss_wait"] >= PATIENCE and state["accuracy_wait"] >= PATIENCE)


def inspect_and_restore():
    """先恢复并验证，再恢复一次随机状态，避免检查消耗续训随机数。"""
    path = SOURCE_DIR / "checkpoint_epoch_45.pt"
    original = json.loads((SOURCE_DIR / "cnn_training_history.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "completed_epoch", "config",
                "random_states", "history", "best_model_state_dict", "model_training"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"完整检查点缺少 {sorted(missing)}，不能声明完全连续恢复。")
    config = checkpoint["config"]
    assert checkpoint["completed_epoch"] == 45 and config == original["config"]
    assert checkpoint["history"]["epochs"] == original["epochs"]
    assert config["learning_rate"] == .03 and config["device"] == "cpu"
    assert config["batch_size"] == BATCH_SIZE == 64 and config["num_workers"] == NUM_WORKERS == 0
    assert config["preprocessing"] == "ToTensor() only"
    assert config["python"] == sys.version.split()[0]
    assert config["versions"] == {name: version(name) for name in config["versions"]}
    assert config["cpu_threads"] == torch.get_num_threads()
    assert not config["test_set_used"]
    for name, digest in original["source_sha256"].items():
        assert hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest() == digest
    rng = checkpoint["random_states"]
    required_rng = {"python", "numpy", "torch_cpu", "train_loader_generator", "validation_loader_generator"}
    if not required_rng.issubset(rng):
        raise RuntimeError(f"缺少随机状态 {sorted(required_rng.difference(rng))}，不能完整恢复。")
    assert all(rng[name] is not None for name in required_rng - {"validation_loader_generator"})
    training, validation = load_training_datasets()  # 只读取原官方训练数据。
    assert index_hash(training.indices) == config["train_indices_sha256"]
    assert index_hash(validation.indices) == config["validation_indices_sha256"]
    assert training.dataset.classes == config["class_names"]
    recording = RecordingDataset(training)
    loader, val_loader = create_train_loader(recording), create_validation_loader(validation)
    model = SimpleCNN().cpu()
    assert str(model) == config["model"]["structure"]
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    assert dict(optimizer.defaults) == config["optimizer_defaults"]
    checkpoint = load_checkpoint(path, model, optimizer, loader, val_loader)
    assert optimizer.state_dict() == checkpoint["optimizer_state_dict"]
    before = model_fingerprint(model)
    expected = original["epochs"][44]["validation"]
    metrics = validate(model, val_loader, nn.CrossEntropyLoss(), torch.device("cpu"), recall_label=6)
    assert metrics == expected
    assert before == model_fingerprint(model) and all(p.grad is None for p in model.parameters())
    # 上面的验证创建了迭代器。恢复原状态，确保下一次取批次仍处于第45轮保存的边界。
    restore_random_states(rng, loader, val_loader)
    assert torch.equal(torch.get_rng_state(), rng["torch_cpu"])
    assert torch.equal(loader.generator.get_state(), rng["train_loader_generator"])
    assert tensor_fingerprint(loader.generator.get_state()) == original["epochs"][-1]["loader_state_after_sha256"]
    assert random.getstate() == rng["python"]
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == rng["numpy"][0] and np.array_equal(actual_numpy[1], rng["numpy"][1])
    assert actual_numpy[2:] == rng["numpy"][2:]
    report = {"can_resume_continuously": True, "source_checkpoint": str(path.relative_to(PROJECT_DIR)),
        "source_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "completed_epoch": 45,
        "restored_parameter_sha256": before, "optimizer_state_restored": True,
        "random_states_restored": sorted(required_rng), "validation_generator": "None; uses saved global torch RNG",
        "validation_matches_epoch45": metrics, "random_state_restored_after_inspection": True,
        "restored_loader_state_sha256": tensor_fingerprint(loader.generator.get_state()), "test_set_used": False}
    print("恢复检查通过：第45轮参数、优化器和随机状态齐全，完整验证指标精确一致。", flush=True)
    print(f"第45轮：损失{metrics['loss']:.6f}，准确率{metrics['accuracy']:.2%}，Shirt召回率{metrics['class_recall']['recall']:.2%}。", flush=True)
    return model, optimizer, loader, val_loader, recording, checkpoint, original, report


def save_continuation_curves(history, output_dir):
    entries = history["epochs"][44:]
    numbers = [e["epoch"] for e in entries]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for split, label in (("train", "Training"), ("validation", "Validation")):
        axes[0].plot(numbers, [e[split]["loss"] for e in entries], marker="o", label=label)
    axes[1].plot(numbers, [100*e["validation"]["accuracy"] for e in entries], marker="o", label="Validation")
    axes[2].plot(numbers, [100*e["validation"]["class_recall"]["recall"] for e in entries], marker="o", label="Shirt")
    for axis, ylabel in zip(axes, ("Mean cross-entropy loss", "Validation accuracy (%)", "Shirt recall (%)")):
        axis.set_xlabel("Total epoch (45 = restored baseline)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(numbers)
        axis.axvline(history["best_epoch"], linestyle="--", color="gray", label="Best accuracy epoch")
        axis.grid(alpha=.3)
        axis.legend()
    figure.suptitle("Continuous CNN training from epoch 45 | SGD lr=0.03 | CPU | patience 8 on both metrics")
    figure.tight_layout()
    figure.savefig(output_dir / "continuation_curves.png", dpi=150, facecolor="white", transparent=False)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-extra-epochs", type=int, choices=range(1, 16), default=15)
    parser.add_argument("--inspect-only", action="store_true", help="只恢复检查，不更新参数或保存结果。")
    args = parser.parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    if not output_dir.is_relative_to((OUTPUT_DIR / "experiments").resolve()):
        parser.error("续训结果必须位于本项目 outputs/experiments 内。")
    if not args.inspect_only and output_dir.exists() and any(output_dir.iterdir()):
        parser.error("输出目录已有内容；请指定新的空目录，禁止覆盖。")
    model, optimizer, loader, val_loader, recording, checkpoint, original, report = inspect_and_restore()
    if args.inspect_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_history(output_dir / "resume_check.json", report)
    history = copy.deepcopy(checkpoint["history"])
    planned_end = 45 + args.max_extra_epochs
    history["original_config"] = copy.deepcopy(history["config"])
    history["config"]["epochs_requested"] = planned_end
    history["epochs_requested"] = planned_end
    history["status"] = "running"
    history["initialization"] = "Continuous resume from final epoch45 model, optimizer and all saved random states"
    history["resumed_from"] = report["source_checkpoint"]
    history["resumed_checkpoint_sha256"] = report["source_checkpoint_sha256"]
    history["source_sha256"]["continue_cnn.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    history["parent_training_seconds"] = original["training_seconds"]
    history["checkpoint_files"] = []
    history.pop("reload_validation", None)
    history.pop("checkpoint_sha256", None)
    baseline = history["epochs"][44]["validation"]
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = baseline
    history["best_epoch"] = 45
    history["best_validation_accuracy"] = baseline["accuracy"]
    history["best_validation_loss"] = baseline["loss"]
    early = {"patience": PATIENCE, "baseline_epoch": 45, "best_loss": baseline["loss"],
             "best_accuracy": baseline["accuracy"], "loss_wait": 0, "accuracy_wait": 0,
             "rule": "Strict improvements since epoch45; stop when both wait counters >=8; ties are not improvements"}
    history["continuation"] = {"start_epoch": 46, "maximum_extra_epochs": args.max_extra_epochs,
                               "completed_extra_epochs": 0, "early_stopping": early}
    best_path, history_path = output_dir / "best_cnn.pt", output_dir / "cnn_training_history.json"
    history["checkpoint"] = str(best_path.relative_to(PROJECT_DIR))
    torch.save(best_state, best_path)  # 无新高时保留起点，不把较差的续训轮次当作最佳。
    write_history(history_path, history)
    criterion, device = nn.CrossEntropyLoss(), torch.device("cpu")
    started = perf_counter()
    stop_reason = "maximum_extra_epochs_reached"
    for epoch in range(46, planned_end + 1):
        print(f"Epoch {epoch}/{planned_end} 开始……", flush=True)
        epoch_started = perf_counter()
        recording.begin_epoch()
        before = tensor_fingerprint(loader.generator.get_state())
        try:
            train_metrics = train_one_epoch(model, loader, criterion, optimizer, device)
            metrics = validate(model, val_loader, criterion, device, recall_label=6)
        except NumericalError as error:
            history.update(status="stopped_numerical_error", failed_epoch=epoch, error=str(error),
                           continuation_seconds=perf_counter()-started)
            write_history(history_path, history)
            print(f"续训停止：{error}；没有重试或修改学习率。", flush=True)
            raise SystemExit(2)
        loss_improved, accuracy_improved, should_stop = update_patience(early, metrics)
        improved = is_better(metrics, best_metrics)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            history["best_epoch"] = epoch
            torch.save(best_state, best_path)
        history["epochs"].append({"epoch": epoch, "train": train_metrics, "validation": metrics,
            "seconds": perf_counter()-epoch_started, "saved_as_best": improved,
            "sample_order": recording.summary(), "loader_state_before_sha256": before,
            "loader_state_after_sha256": tensor_fingerprint(loader.generator.get_state()),
            "validation_loss_improved": loss_improved, "validation_accuracy_new_high": accuracy_improved,
            "loss_wait": early["loss_wait"], "accuracy_wait": early["accuracy_wait"]})
        history["best_validation_accuracy"] = best_metrics["accuracy"]
        history["best_validation_loss"] = best_metrics["loss"]
        history["continuation"]["completed_extra_epochs"] = epoch - 45
        history["continuation_seconds"] = perf_counter()-started
        history["training_seconds"] = original["training_seconds"] + history["continuation_seconds"]
        if should_stop:
            stop_reason = "both_metrics_no_improvement_for_8_epochs"
        history["continuation"]["stop_reason"] = stop_reason if should_stop or epoch == planned_end else None
        filename = f"checkpoint_epoch_{epoch:02d}.pt"
        history["checkpoint_files"].append(filename)
        torch.save(make_checkpoint(model, optimizer, loader, val_loader, history, best_state), output_dir / filename)
        history["final_checkpoint"] = str((output_dir / filename).relative_to(PROJECT_DIR))
        write_history(history_path, history)
        print(f"Epoch {epoch} | 训练损失{train_metrics['loss']:.6f} | 验证损失{metrics['loss']:.6f}，"
              f"准确率{metrics['accuracy']:.2%}，Shirt召回率{metrics['class_recall']['recall']:.2%} | "
              f"损失等待{early['loss_wait']}/8，准确率等待{early['accuracy_wait']}/8 | "
              f"耗时{history['epochs'][-1]['seconds']:.2f}s{' | 保存最佳' if improved else ''}", flush=True)
        if should_stop:
            print("两项指标均连续至少8轮未改善，按约定提前停止。", flush=True)
            break
    history["continuation_seconds"] = perf_counter()-started
    history["training_seconds"] = original["training_seconds"] + history["continuation_seconds"]
    history["continuation_parameter_updates"] = sum(e["train"]["batches"] for e in history["epochs"][45:])
    history["total_parameter_updates"] = sum(e["train"]["batches"] for e in history["epochs"])
    reloaded = SimpleCNN().cpu()
    reloaded.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    before = model_fingerprint(reloaded)
    metrics = validate(reloaded, val_loader, criterion, device, recall_label=6)
    assert metrics == best_metrics and before == model_fingerprint(reloaded)
    assert all(p.grad is None for p in reloaded.parameters())
    history.update(status="completed", reload_validation=metrics,
                   checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest())
    save_continuation_curves(history, output_dir)
    write_history(history_path, history)
    print(f"续训完成：新增{history['continuation']['completed_extra_epochs']}轮，停止原因{stop_reason}，"
          f"最佳第{history['best_epoch']}轮，准确率{metrics['accuracy']:.2%}，损失{metrics['loss']:.6f}，"
          f"Shirt召回率{metrics['class_recall']['recall']:.2%}；实际续训耗时{history['continuation_seconds']:.2f}s。", flush=True)


if __name__ == "__main__":
    main()
