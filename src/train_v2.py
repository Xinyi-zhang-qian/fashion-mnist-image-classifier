"""V2：复用已验证的 CNN 训练函数，独立训练 45 轮并保存可恢复检查点。"""

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from cnn_model import SimpleCNN
from data import (BATCH_SIZE, NUM_WORKERS, OUTPUT_DIR, PROJECT_DIR, SEED,
                  create_train_loader, create_validation_loader, load_training_datasets)
from train import (NumericalError, check_flow, model_fingerprint, tensor_fingerprint,
                   train_one_epoch, validate, write_history)

EPOCHS = 45
MILESTONES = (15, 30, 45)
BEST_RULE = "Higher validation accuracy; ties prefer lower validation loss; exact ties keep earlier."


def index_hash(indices):
    """固定为小端 int64，哈希反映样本顺序，不仅是生成器状态。"""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


class RecordingDataset(Dataset):
    """只记录实际读取的官方训练索引，返回的图片与标签保持原样。"""
    def __init__(self, subset):
        self.subset = subset
        self.indices = subset.indices
        self.seen = []

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        self.seen.append(int(self.indices[index]))
        return self.subset[index]

    def begin_epoch(self):
        self.seen.clear()

    def summary(self):
        assert len(self.seen) == len(self)
        assert sorted(self.seen) == sorted(self.indices)
        return {"sha256": index_hash(self.seen), "samples": len(self.seen),
                "unique_samples": len(set(self.seen)), "encoding": "official index, little-endian int64",
                "first_16": self.seen[:16], "last_16": self.seen[-16:]}


def reset_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def capture_random_states(train_loader, validation_loader):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "train_loader_generator": train_loader.generator.get_state(),
            "validation_loader_generator": (validation_loader.generator.get_state()
                if validation_loader.generator is not None else None)}


def restore_random_states(states, train_loader, validation_loader):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    train_loader.generator.set_state(states["train_loader_generator"])
    if states["validation_loader_generator"] is not None:
        validation_loader.generator.set_state(states["validation_loader_generator"])


def make_checkpoint(model, optimizer, train_loader, validation_loader, history, best_state):
    """轮次结束、验证完成后保存；恢复从下一轮开始。"""
    return {"format_version": 1, "completed_epoch": history["epochs"][-1]["epoch"],
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "config": copy.deepcopy(history["config"]), "history": copy.deepcopy(history),
            "best_model_state_dict": best_state,
            "random_states": capture_random_states(train_loader, validation_loader),
            "model_training": model.training, "resume_boundary": "after full validation, before next epoch"}


def load_checkpoint(path, model, optimizer, train_loader, validation_loader):
    # 完整检查点含 Python/NumPy 状态，只读取本项目自己生成的可信文件。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == 1
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train(checkpoint["model_training"])
    # 新建对象会消耗随机数，所以所有对象创建及参数加载结束后才恢复随机状态。
    restore_random_states(checkpoint["random_states"], train_loader, validation_loader)
    return checkpoint


def is_better(metrics, best):
    return best is None or (metrics["accuracy"], -metrics["loss"]) > (best["accuracy"], -best["loss"])


def check_resume(training, validation):
    """临时小数据、临时模型：比较连续两轮与保存后恢复的第二轮，之后全部丢弃。"""
    small_train = Subset(training.dataset, list(training.indices[:192]))
    small_val = Subset(validation.dataset, list(validation.indices[:128]))
    reset_seeds()
    model = SimpleCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    recorded = RecordingDataset(small_train)
    loader, val_loader = create_train_loader(recorded), create_validation_loader(small_val)
    criterion, device = nn.CrossEntropyLoss(), torch.device("cpu")
    first = train_one_epoch(model, loader, criterion, optimizer, device)
    val_first = validate(model, val_loader, criterion, device, recall_label=6)
    history = {"config": {"preflight": True}, "epochs": [{"epoch": 1, "train": first, "validation": val_first}]}
    with tempfile.TemporaryDirectory(prefix="ai_classifier_resume_") as temporary:
        path = Path(temporary) / "checkpoint.pt"
        torch.save(make_checkpoint(model, optimizer, loader, val_loader, history,
                                   copy.deepcopy(model.state_dict())), path)
        recorded.begin_epoch()
        expected_train = train_one_epoch(model, loader, criterion, optimizer, device)
        expected_val = validate(model, val_loader, criterion, device, recall_label=6)
        expected_order = recorded.summary()
        expected_parameters = copy.deepcopy(model.state_dict())
        expected_generator = loader.generator.get_state().clone()
        expected_draws = (random.random(), np.random.random(), torch.rand(3))
        reset_seeds()
        restored = SimpleCNN()
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
        restored_dataset = RecordingDataset(small_train)
        restored_loader = create_train_loader(restored_dataset)
        restored_val_loader = create_validation_loader(small_val)
        checkpoint = load_checkpoint(path, restored, restored_optimizer, restored_loader, restored_val_loader)
        assert checkpoint["completed_epoch"] == 1
        actual_train = train_one_epoch(restored, restored_loader, criterion, restored_optimizer, device)
        actual_val = validate(restored, restored_val_loader, criterion, device, recall_label=6)
        actual_draws = (random.random(), np.random.random(), torch.rand(3))
        assert actual_train == expected_train and actual_val == expected_val
        assert restored_dataset.summary() == expected_order
        assert torch.equal(restored_loader.generator.get_state(), expected_generator)
        assert all(torch.equal(value, restored.state_dict()[name]) for name, value in expected_parameters.items())
        assert actual_draws[:2] == expected_draws[:2] and torch.equal(actual_draws[2], expected_draws[2])
    assert is_better({"accuracy": .8, "loss": .4}, {"accuracy": .8, "loss": .5})
    assert not is_better({"accuracy": .8, "loss": .5}, {"accuracy": .8, "loss": .5})
    print("恢复检查通过：参数、第二轮指标、实际样本顺序和三种随机状态完全一致；平分规则通过。", flush=True)
    return {"temporary_model": True, "resume_exact_parameters": True,
            "resume_exact_metrics_and_order": True, "python_numpy_torch_rng_restored": True,
            "best_tie_rule_checked": True, "test_set_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, choices=(0.01, 0.03), default=0.01)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", type=Path, help="本项目完整检查点；写入新的空目录，恢复至第45轮。")
    args = parser.parse_args()
    output_dir = (PROJECT_DIR / (args.output_dir or Path(
        f"outputs/experiments/v2_sgd_lr{'001' if args.lr == 0.01 else '003'}_45epochs"))).resolve()
    if not output_dir.is_relative_to((OUTPUT_DIR / "experiments").resolve()):
        parser.error("V2 必须保存到本项目 outputs/experiments 内。")
    if not args.check_only and output_dir.exists() and any(output_dir.iterdir()):
        parser.error("结果目录已有内容；为保留结果，请使用新的 --output-dir，不会覆盖。")
    print(f"解释器：{sys.executable}；CPU；SGD lr={args.lr}；固定45轮，batch={BATCH_SIZE}。", flush=True)
    training, validation = load_training_datasets()
    assert len(training) == 50000 and len(validation) == 10000
    assert not set(training.indices).intersection(validation.indices)
    if args.check_only:
        checks = {}
        for rate in (0.01, 0.03):
            checks[str(rate)] = check_flow(training, validation, torch.device("cpu"), SimpleCNN, rate)
        checks["resume"] = check_resume(training, validation)
        audit_dir = OUTPUT_DIR / "experiments/v2_comparison"
        audit_dir.mkdir(parents=True, exist_ok=True)
        write_history(audit_dir / "preflight_checks.json", checks)
        return
    # 每次正式实验都重新初始化，不复用流程检查或旧模型。
    reset_seeds()
    model = SimpleCNN().cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion, device = nn.CrossEntropyLoss(), torch.device("cpu")
    recording = RecordingDataset(training)
    train_loader, val_loader = create_train_loader(recording), create_validation_loader(validation)
    config = {
        "model": {"name": "SimpleCNN", "structure": str(model), "parameter_count": sum(p.numel() for p in model.parameters())},
        "learning_rate": args.lr, "optimizer": "SGD (momentum=0, weight_decay=0)",
        "optimizer_defaults": dict(optimizer.defaults), "loss_function": "CrossEntropyLoss (raw logits, reduction=mean)",
        "seed": SEED, "split_seed": SEED, "shuffle_seed": SEED,
        "preprocessing": "ToTensor() only", "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
        "train_shuffle": True, "validation_shuffle": False, "drop_last": False,
        "device": "cpu", "cpu_threads": torch.get_num_threads(), "epochs_requested": EPOCHS,
        "train_samples": len(training), "validation_samples": len(validation),
        "batches_per_epoch": len(train_loader), "validation_batches": len(val_loader),
        "train_indices_sha256": index_hash(training.indices), "validation_indices_sha256": index_hash(validation.indices),
        "class_names": training.dataset.classes, "scheduler": None, "augmentation": None,
        "python": sys.version.split()[0],
        "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "matplotlib")},
        "test_set_used": False, "best_selection_rule": BEST_RULE,
    }
    history = {**copy.deepcopy(config), "config": config, "status": "running", "epochs": [],
        "initialization": "Fresh model and optimizer; fresh training generator seeded 42",
        "initial_parameter_sha256": model_fingerprint(model),
        "initial_loader_state_sha256": tensor_fingerprint(train_loader.generator.get_state()),
        "source_sha256": {name: hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest()
                          for name in ("cnn_model.py", "data.py", "train.py", "train_v2.py")},
        "checkpoint_files": [], "best_epoch": None, "training_seconds": 0,
        "numerical_checks": "Reused finite logits/loss/gradient/parameter checks; no clipping"}
    best_state, best_metrics, start_epoch = None, None, 1
    if args.resume:
        resume_path = (PROJECT_DIR / args.resume).resolve()
        if not resume_path.is_relative_to(OUTPUT_DIR.resolve()):
            parser.error("只允许恢复本项目自己生成的检查点。")
        checkpoint = load_checkpoint(resume_path, model, optimizer, train_loader, val_loader)
        assert checkpoint["config"] == config, "恢复配置必须完全一致，包括学习率。"
        history = checkpoint["history"]
        assert history["source_sha256"] == {name: hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest()
                                             for name in history["source_sha256"]}
        best_state = checkpoint["best_model_state_dict"]
        best_metrics = history["epochs"][history["best_epoch"] - 1]["validation"]
        start_epoch = checkpoint["completed_epoch"] + 1
        history["resumed_from"] = str(resume_path)
        history["checkpoint_files"] = []
        print(f"恢复已完成第{start_epoch - 1}轮的状态；从第{start_epoch}轮继续至45轮。", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path, best_path = output_dir / "cnn_training_history.json", output_dir / "best_cnn.pt"
    history["checkpoint"] = str(best_path.relative_to(PROJECT_DIR))
    if best_state is not None:
        torch.save(best_state, best_path)
    write_history(history_path, history)
    started, previous_seconds = perf_counter(), history["training_seconds"]
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} 开始……", flush=True)
        epoch_started = perf_counter()
        recording.begin_epoch()
        before = tensor_fingerprint(train_loader.generator.get_state())
        try:
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            validation_metrics = validate(model, val_loader, criterion, device, recall_label=6)
        except NumericalError as error:
            history.update(status="stopped_numerical_error", failed_epoch=epoch, error=str(error),
                           training_seconds=previous_seconds + perf_counter() - started)
            write_history(history_path, history)
            print(f"该实验已停止：{error}；保留已完成轮次。", flush=True)
            raise SystemExit(2)
        order = recording.summary()
        assert train_metrics["samples"] == 50000 and validation_metrics["samples"] == 10000
        improved = is_better(validation_metrics, best_metrics)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)
            best_metrics = validation_metrics
            history["best_epoch"] = epoch
        history["epochs"].append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics,
            "seconds": perf_counter() - epoch_started, "saved_as_best": improved,
            "sample_order": order, "loader_state_before_sha256": before,
            "loader_state_after_sha256": tensor_fingerprint(train_loader.generator.get_state())})
        history["best_validation_accuracy"] = best_metrics["accuracy"]
        history["best_validation_loss"] = best_metrics["loss"]
        history["training_seconds"] = previous_seconds + perf_counter() - started
        if epoch in MILESTONES:
            name = f"checkpoint_epoch_{epoch:02d}.pt"
            history["checkpoint_files"].append(name)
            torch.save(make_checkpoint(model, optimizer, train_loader, val_loader, history, best_state), output_dir / name)
        write_history(history_path, history)
        print(f"Epoch {epoch}/{EPOCHS} | 训练损失 {train_metrics['loss']:.6f}，准确率 {train_metrics['accuracy']:.2%} | "
              f"验证损失 {validation_metrics['loss']:.6f}，准确率 {validation_metrics['accuracy']:.2%} | "
              f"Shirt召回率 {validation_metrics['class_recall']['recall']:.2%} | "
              f"耗时 {history['epochs'][-1]['seconds']:.2f}s{' | 保存最佳' if improved else ''}", flush=True)
    history["training_seconds"] = previous_seconds + perf_counter() - started
    history["total_parameter_updates"] = sum(e["train"]["batches"] for e in history["epochs"])
    # 正式训练已结束。重新建立模型并核对同一个最佳轮次，不拼接指标。
    reloaded = SimpleCNN().cpu()
    reloaded.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    parameter_hash = model_fingerprint(reloaded)
    metrics = validate(reloaded, val_loader, criterion, device, recall_label=6)
    assert metrics == best_metrics
    assert model_fingerprint(reloaded) == parameter_hash and all(p.grad is None for p in reloaded.parameters())
    history.update(status="completed", reload_validation=metrics,
                   checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest())
    write_history(history_path, history)
    print(f"重新加载核对通过：最佳第{history['best_epoch']}轮，验证准确率{metrics['accuracy']:.2%}，"
          f"损失{metrics['loss']:.6f}；训练耗时{history['training_seconds']:.2f}s，"
          f"更新{history['total_parameter_updates']}次。", flush=True)


if __name__ == "__main__":
    main()
