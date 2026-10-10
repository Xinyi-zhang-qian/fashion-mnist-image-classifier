# 独立发布与GitHub上传

原工作目录属于一个包含其他项目的父Git仓库。发布版导出到独立的
`releases/fashion-mnist-image-classifier/`，只在导出目录初始化Git；不操作父仓库索引或历史。

## 发布内容

仅包含代码、依赖说明、学习记录、实验配置/指标、图表、固定最终模型和文件清单。
数据集、缓存、临时文件、旧模型权重及逐轮检查点不进入发布版。
工作站个人路径只在公开副本中替换；原始实验文件保留原哈希。

最终权重426,764字节，已随Git仓库和ZIP提供，无需单独发布权重。
`release_contents.json`记录每个文件的公开副本与原始文件SHA-256；
`final_model/SHA256SUMS`核对推理权重和配置。

## 从原工作目录重新导出

```bash
python src/build_release.py --export-dir ../../releases/fashion-mnist-image-classifier-repeat --zip dist/fashion-mnist-image-classifier-repeat.zip
```

指定的发布目录和ZIP必须尚不存在或目录为空。脚本不训练、不重新评估测试集。

## 本地提交后上传

在独立发布目录执行以下命令。需要本机GitHub CLI完成登录，勿把访问令牌写入项目文件。

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth status
gh repo create fashion-mnist-image-classifier --public --source . --remote origin --push
gh repo view --web
```

如同名仓库已存在，不覆盖或复用他人仓库；可改用
`fashion-mnist-image-classifier-20261010`等合理新名称。

检查公开状态、远端默认分支及README：

```bash
gh repo view --json nameWithOwner,url,isPrivate,defaultBranchRef
git ls-remote origin refs/heads/main
gh api repos/{owner}/{repo}/readme --jq .html_url
```

ZIP可以作为项目Release附件上传，也可直接使用GitHub提供的源码ZIP；
源码ZIP包含固定最终权重，模型下载不依赖其他训练检查点。
