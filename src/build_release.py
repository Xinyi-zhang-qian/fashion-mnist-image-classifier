"""按允许清单导出独立发布目录和ZIP，不携带数据集、缓存或逐轮检查点。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from data import PROJECT_DIR


REPOSITORY_NAME = "fashion-mnist-image-classifier"
TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".yml", ".csv"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)


def release_files():
    files = [PROJECT_DIR / name for name in
             ("README.md", "PROGRESS.md", "requirements.txt", "environment.yml", ".gitignore")]
    files.extend(sorted((PROJECT_DIR / "src").glob("*.py")))
    files.extend(sorted((PROJECT_DIR / "docs").glob("*.md")))
    files.extend(sorted((PROJECT_DIR / "docs").glob("*.json")))
    files.extend(sorted((PROJECT_DIR / "final_model").iterdir()))
    for path in sorted((PROJECT_DIR / "outputs").rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".png", ".csv"}:
            continue
        if any(part in {".matplotlib", "__pycache__", "source"} for part in path.parts):
            continue
        if path.name.endswith("_validation_predictions.json") or path.name.startswith("final_prediction"):
            continue
        files.append(path)
    return sorted(set(files))


def public_bytes(path):
    content = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "SHA256SUMS"}:
        text = content.decode("utf-8")
        # 仅发布副本替换工作站绝对路径，原始实验文件保持不变。
        if path.suffix != ".py":
            text = text.replace(str(PROJECT_DIR), REPOSITORY_NAME)
            text = text.replace("/opt/anaconda3/envs/ai-classifier/bin/python", "python")
        if re.search(r"/" + r"(?:Users|home)/[^/\s]+/", text):
            raise ValueError(f"发布副本仍有个人路径：{path.relative_to(PROJECT_DIR)}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValueError(f"发现疑似凭据，停止打包：{path.relative_to(PROJECT_DIR)}")
        content = text.encode("utf-8")
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True, help="新的独立发布目录，不在原项目内部。")
    parser.add_argument("--zip", type=Path, default=PROJECT_DIR / "dist" / f"{REPOSITORY_NAME}.zip")
    args = parser.parse_args()
    destination = args.export_dir.resolve()
    zip_path = args.zip.resolve()
    if destination == PROJECT_DIR or destination.is_relative_to(PROJECT_DIR):
        parser.error("发布目录必须在原项目之外，避免在原项目创建嵌套仓库。")
    if destination.exists() and any(destination.iterdir()):
        parser.error("发布目录已有文件，不自动覆盖。")
    if zip_path.exists():
        parser.error("ZIP已存在，请指定新的文件名，不自动覆盖。")

    # 先完成内容检查，再写出；除final_model/model.pt外不允许模型二进制文件。
    payloads, manifest = {}, []
    for path in release_files():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"不允许缺失文件、目录或符号链接：{path.name}")
        relative = path.relative_to(PROJECT_DIR)
        if path.suffix == ".pt" and str(relative) != "final_model/model.pt":
            raise ValueError(f"未批准的权重：{relative}")
        content = public_bytes(path)
        if len(content) > 5 * 1024 * 1024:
            raise ValueError(f"单文件超过本发布包保守上限5MiB：{relative}")
        payloads[relative] = content
        manifest.append({"path": str(relative), "bytes": len(content),
                         "sha256": hashlib.sha256(content).hexdigest(),
                         "original_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert Path("final_model/model.pt") in payloads
    weight_hash = hashlib.sha256(payloads[Path("final_model/model.pt")]).hexdigest()
    assert weight_hash == "c972ca96c5d63c81998a6d884f57ebc42406f04a02c5368bdf1d01178d19a570"
    package_manifest = {
        "repository_name": REPOSITORY_NAME, "file_count_excluding_this_manifest": len(manifest),
        "files": manifest,
        "public_copy_changes": "仅替换文档/元数据中的工作站绝对路径；原实验不修改；代码与权重保持原字节。",
        "exclusions": ["data/", ".git/", "缓存与系统文件", "逐轮checkpoint_epoch_*.pt",
                       "其他历史模型权重", "逐图验证预测", "运行时预测图", "dist/"],
        "test_re_evaluated": False,
    }
    payloads[Path("release_contents.json")] = (json.dumps(package_manifest, ensure_ascii=False,
                                                         indent=2, allow_nan=False) + "\n").encode("utf-8")
    destination.mkdir(parents=True, exist_ok=True)
    for relative, content in payloads.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(payloads):
            archive.write(destination / relative, f"{REPOSITORY_NAME}/{relative}")
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert archive.read(f"{REPOSITORY_NAME}/final_model/model.pt") == payloads[Path("final_model/model.pt")]
    print(f"发布目录：{destination}\n文件数：{len(payloads)}\nZIP：{zip_path}")
    print(f"ZIP大小：{zip_path.stat().st_size}字节\nZIP SHA-256：{hashlib.sha256(zip_path.read_bytes()).hexdigest()}")
    print("允许清单、凭据模式、个人路径与ZIP完整性检查通过。")


if __name__ == "__main__":
    main()
