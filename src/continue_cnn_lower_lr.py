"""从 V2 第45轮完整状态恢复，将SGD学习率改为0.01，完整训练46—60轮。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch
from torch import nn

from cnn_model import SimpleCNN
from continue_cnn import inspect_and_restore
from data import OUTPUT_DIR, PROJECT_DIR
from train import NumericalError, model_fingerprint, tensor_fingerprint, train_one_epoch, validate, write_history
from train_v2 import is_better, make_checkpoint

import matplotlib.pyplot as plt

OUTPUT = OUTPUT_DIR / "experiments/v2_resume45_lr001_15epochs"
PREVIOUS = OUTPUT_DIR / "experiments/v2_sgd_lr003_resume45_max15epochs"


def lower_learning_rate(optimizer):
    """先由已有恢复函数加载优化器状态，此处只改学习率。"""
    before = copy.deepcopy(optimizer.state_dict())
    defaults_before = dict(optimizer.defaults)
    assert all(group["lr"] == .03 for group in optimizer.param_groups)
    for group in optimizer.param_groups:
        group["lr"] = .01
    optimizer.defaults["lr"] = .01
    after = copy.deepcopy(optimizer.state_dict())
    expected = copy.deepcopy(before)
    for group in expected["param_groups"]:
        group["lr"] = .01
    assert after == expected  # 优化器其他设置和状态未改。
    assert dict(optimizer.defaults) == {**defaults_before, "lr": .01}
    return {"group_lrs_after_load": [g["lr"] for g in before["param_groups"]],
            "group_lrs_after_change": [g["lr"] for g in after["param_groups"]],
            "other_optimizer_settings_and_state_unchanged": True,
            "actual_optimizer_groups": after["param_groups"]}


def compare_and_plot(history, output_dir):
    """只读取已有0.03续训；比较共同轮次，并保留不同预算的限制。"""
    previous = json.loads((PREVIOUS / "cnn_training_history.json").read_text(encoding="utf-8"))
    assert history["resumed_checkpoint_sha256"] == previous["resumed_checkpoint_sha256"]
    common_end = min(len(history["epochs"]), len(previous["epochs"]))
    for a, b in zip(history["epochs"][45:common_end], previous["epochs"][45:common_end]):
        assert a["sample_order"] == b["sample_order"]
        assert a["loader_state_before_sha256"] == b["loader_state_before_sha256"]
        assert a["loader_state_after_sha256"] == b["loader_state_after_sha256"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    specifications = (
        ("train", "loss", "Training loss", 1),
        ("validation", "loss", "Validation loss", 1),
        ("validation", "accuracy", "Validation accuracy (%)", 100),
        ("validation", "shirt", "Shirt validation recall (%)", 100),
    )
    for data, label, style in ((history, "lr=0.01, full 15 extra epochs", "-"),
                               (previous, "lr=0.03, stopped at epoch 54", "--")):
        entries = data["epochs"][44:]
        numbers = [e["epoch"] for e in entries]
        for axis, (split, key, ylabel, scale) in zip(axes.flat, specifications):
            values = [(e[split]["class_recall"]["recall"] if key == "shirt" else e[split][key]) * scale
                      for e in entries]
            axis.plot(numbers, values, linestyle=style, marker="o", markersize=3, label=label)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Total epoch (45 = shared restored state)")
            axis.set_xticks(range(45, 61))
            axis.grid(alpha=.3)
            axis.legend(fontsize=9)
    figure.suptitle("CNN continuation from the same epoch 45 | CPU | only learning rate changes\n"
                    "Prior lr=0.03 branch has no observations after epoch 54")
    figure.tight_layout()
    figure.savefig(output_dir / "learning_rate_continuation_comparison.png", dpi=150,
                   facecolor="white", transparent=False)
    plt.close(figure)
    selected = history["epochs"][history["best_epoch"] - 1]
    common = {}
    for data, name in ((history, "lr001"), (previous, "lr003")):
        entries = data["epochs"][45:common_end]
        best = max(data["epochs"][44:common_end],
                   key=lambda e: (e["validation"]["accuracy"], -e["validation"]["loss"], -e["epoch"]))
        common[name] = {"best_epoch_including_baseline": best["epoch"], "best_validation": best["validation"],
            "mean_training_loss": fmean(e["train"]["loss"] for e in entries),
            "mean_validation_loss": fmean(e["validation"]["loss"] for e in entries),
            "mean_validation_accuracy": fmean(e["validation"]["accuracy"] for e in entries),
            "mean_shirt_recall": fmean(e["validation"]["class_recall"]["recall"] for e in entries)}
    summary = {"source_epoch": 45, "source_checkpoint": history["resumed_from"],
        "source_checkpoint_sha256": history["resumed_checkpoint_sha256"],
        "baseline_validation": history["epochs"][44]["validation"],
        "selected_epoch": selected["epoch"], "selected_validation": selected["validation"],
        "selected_checkpoint": history["checkpoint"], "selected_checkpoint_sha256": history["checkpoint_sha256"],
        "final_epoch": 60, "final_validation": history["epochs"][-1]["validation"],
        "accuracy_change_from_baseline_percentage_points": 100 * (
            selected["validation"]["accuracy"] - history["epochs"][44]["validation"]["accuracy"]),
        "common_continuation_epochs": [46, common_end], "matched_sample_order_epochs": common_end - 45,
        "common_window": common, "completed_extra_epochs": 15,
        "continuation_seconds": history["continuation_seconds"], "test_set_used": False,
        "limitations": ["The old 0.03 branch stopped at 54; only 46-54 is a matched-budget comparison.",
                        "One checkpoint, one validation split and one random trajectory; no test-set evidence.",
                        "Selected loss, accuracy and Shirt recall all belong to the same checkpoint."]}
    write_history(output_dir / "learning_rate_continuation_comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--inspect-only", action="store_true", help="只恢复并核实学习率修改，不更新参数、不保存结果。")
    args = parser.parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    if not output_dir.is_relative_to((OUTPUT_DIR / "experiments").resolve()):
        parser.error("实验目录必须位于本项目 outputs/experiments 内。")
    if not args.inspect_only and output_dir.exists() and any(output_dir.iterdir()):
        parser.error("目录已有结果，请指定新的空目录；不会覆盖。")
    model, optimizer, loader, val_loader, recording, checkpoint, original, report = inspect_and_restore()
    fingerprint, rng_before = model_fingerprint(model), torch.get_rng_state().clone()
    report["learning_rate_change"] = lower_learning_rate(optimizer)
    assert fingerprint == model_fingerprint(model) and torch.equal(rng_before, torch.get_rng_state())
    assert torch.equal(loader.generator.get_state(), checkpoint["random_states"]["train_loader_generator"])
    print(f"优化器加载后学习率{report['learning_rate_change']['group_lrs_after_load']}，"
          f"修改后实际参数组学习率{[g['lr'] for g in optimizer.param_groups]}；其他设置不变。", flush=True)
    if args.inspect_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_history(output_dir / "resume_check.json", report)
    history = copy.deepcopy(checkpoint["history"])
    history["original_config"] = copy.deepcopy(history["config"])
    history["config"].update(learning_rate=.01, epochs_requested=60,
                            optimizer_defaults=dict(optimizer.defaults), early_stopping=False)
    history.update(learning_rate=.01, epochs_requested=60, optimizer_defaults=dict(optimizer.defaults),
        status="running", initialization="Full-state resume from epoch45; optimizer loaded before lr change",
        resumed_from=report["source_checkpoint"], resumed_checkpoint_sha256=report["source_checkpoint_sha256"],
        parent_training_seconds=original["training_seconds"], checkpoint_files=[],
        learning_rate_schedule=[{"epochs": [1, 45], "learning_rate": .03},
                                {"epochs": [46, 60], "learning_rate": .01}],
        continuation={"start_epoch": 46, "extra_epochs_requested": 15,
                      "completed_extra_epochs": 0, "early_stopping": False})
    history["source_sha256"].update({name: hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest()
                                    for name in ("continue_cnn.py", "continue_cnn_lower_lr.py")})
    history.pop("reload_validation", None)
    history.pop("checkpoint_sha256", None)
    best_state, best_metrics = copy.deepcopy(model.state_dict()), history["epochs"][44]["validation"]
    history.update(best_epoch=45, best_validation_accuracy=best_metrics["accuracy"],
                   best_validation_loss=best_metrics["loss"])
    best_path, history_path = output_dir / "best_cnn.pt", output_dir / "cnn_training_history.json"
    history["checkpoint"] = str(best_path.relative_to(PROJECT_DIR))
    torch.save(best_state, best_path)
    write_history(history_path, history)
    criterion, device = nn.CrossEntropyLoss(), torch.device("cpu")
    started = perf_counter()
    for epoch in range(46, 61):
        print(f"Epoch {epoch}/60 开始……（不早停）", flush=True)
        epoch_started = perf_counter()
        recording.begin_epoch()
        before = tensor_fingerprint(loader.generator.get_state())
        assert all(g["lr"] == .01 for g in optimizer.param_groups)
        try:
            train_metrics = train_one_epoch(model, loader, criterion, optimizer, device)
            metrics = validate(model, val_loader, criterion, device, recall_label=6)
        except NumericalError as error:
            history.update(status="stopped_numerical_error", failed_epoch=epoch, error=str(error),
                           continuation_seconds=perf_counter() - started)
            write_history(history_path, history)
            print(f"异常停止：{error}；记录已完成轮次，不修改配置重试。", flush=True)
            raise SystemExit(2)
        assert train_metrics["samples"] == 50000 and metrics["samples"] == 10000
        improved = is_better(metrics, best_metrics)
        if improved:
            best_state, best_metrics = copy.deepcopy(model.state_dict()), metrics
            history["best_epoch"] = epoch
            torch.save(best_state, best_path)
        history["epochs"].append({"epoch": epoch, "train": train_metrics, "validation": metrics,
            "seconds": perf_counter() - epoch_started, "saved_as_best": improved, "learning_rate": .01,
            "optimizer_group_lrs": [g["lr"] for g in optimizer.param_groups],
            "sample_order": recording.summary(), "loader_state_before_sha256": before,
            "loader_state_after_sha256": tensor_fingerprint(loader.generator.get_state())})
        history.update(best_validation_accuracy=best_metrics["accuracy"], best_validation_loss=best_metrics["loss"],
                       continuation_seconds=perf_counter() - started)
        history["training_seconds"] = original["training_seconds"] + history["continuation_seconds"]
        history["continuation"]["completed_extra_epochs"] = epoch - 45
        filename = f"checkpoint_epoch_{epoch:02d}.pt"
        history["checkpoint_files"].append(filename)
        history["final_checkpoint"] = str((output_dir / filename).relative_to(PROJECT_DIR))
        torch.save(make_checkpoint(model, optimizer, loader, val_loader, history, best_state), output_dir / filename)
        write_history(history_path, history)
        print(f"Epoch {epoch} | lr={[g['lr'] for g in optimizer.param_groups]} | "
              f"训练损失{train_metrics['loss']:.6f} | 验证损失{metrics['loss']:.6f}，准确率{metrics['accuracy']:.2%}，"
              f"Shirt召回率{metrics['class_recall']['recall']:.2%} | "
              f"耗时{history['epochs'][-1]['seconds']:.2f}s{' | 保存最佳' if improved else ''}", flush=True)
    history.update(continuation_seconds=perf_counter() - started,
        continuation_parameter_updates=sum(e["train"]["batches"] for e in history["epochs"][45:]),
        total_parameter_updates=sum(e["train"]["batches"] for e in history["epochs"]))
    history["training_seconds"] = original["training_seconds"] + history["continuation_seconds"]
    reloaded = SimpleCNN().cpu()
    reloaded.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    before = model_fingerprint(reloaded)
    metrics = validate(reloaded, val_loader, criterion, device, recall_label=6)
    assert metrics == best_metrics and before == model_fingerprint(reloaded)
    assert all(p.grad is None for p in reloaded.parameters())
    history.update(status="completed", reload_validation=metrics,
                   checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest())
    write_history(history_path, history)
    comparison = compare_and_plot(history, output_dir)
    print(f"完整15轮完成；最佳第{history['best_epoch']}轮，验证准确率{metrics['accuracy']:.2%}，"
          f"损失{metrics['loss']:.6f}，Shirt召回率{metrics['class_recall']['recall']:.2%}。", flush=True)
    print(f"起点准确率变化{comparison['accuracy_change_from_baseline_percentage_points']:+.2f}个百分点；"
          f"实际续训耗时{history['continuation_seconds']:.2f}s，更新{history['continuation_parameter_updates']}次。", flush=True)


if __name__ == "__main__":
    main()
