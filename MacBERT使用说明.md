
# MacBERT 影评对照实验

双击 启动系统.cmd，网页默认使用本地 MacBERT。可在页首切换为 TF-IDF 增强版，单条预测、作品看板和实验页随当前选择切换。每个模型的批量结果在当前会话内独立保存，两份下载对应当前选择。也可双击 MacBERT评分.cmd，在命令窗口快速测试一条评论。

## 查看评测

- reports/macbert/MacBERT对照实测报告.md 提供新增作品、原测试集上的两个模型实测指标。
- reports/macbert/metrics.json 提供固定划分、验证集选择、逐类指标和混淆矩阵。
- reports/macbert 下的 test_predictions.csv 和 legacy_predictions.csv 记录逐条新模型预测，movies 文件记录作品汇总。

## 批量评分

在 PowerShell 中进入项目目录，执行：

    .\MacBERT评分.cmd --csv "待评分.csv" --out "预测结果"

输入采用 UTF-8，必需列为 text。可提供 movie_id 和 movie_name 用于按作品聚合。两份输出为 review_predictions.csv 和 movie_scores.csv。

## 训练记录

- 模型来自 HFL 的 chinese-macbert-base。原权重来自 ModelScope 镜像，并按原发布页做 SHA-256 校验。
- 训练集有 25,648 条，验证集 4,504 条，新增测试集 4,749 条，历史测试集 945 条。训练、验证和测试以作品隔离。
- 冻结嵌入层和前六个 Transformer 层，微调后六层及情感、五星评分两任务输出头。两轮训练，输入最多 96 个词元，长短评论按长度组批。验证集决定情感权重、星级映射及检查点。
- 该机使用 Anaconda llm 环境中的 Python 3.11 与 PyTorch 2.5.1 CUDA 12.1。项目的 .bert_dependencies 存放 Transformers 4.51.3 及配套库。模型保存在 models/macbert，推理可在 CPU 上运行。
- 网页使用 .bert_web_dependencies 中的 Streamlit 与项目固定依赖。启动脚本自动选择本机的 llm 环境。基础 .venv 环境仍可运行 TF-IDF 网页。

复训需要原预训练 checkpoint 和分组数据。将三个官方模型文件 config.json、vocab.txt、pytorch_model.bin 放在 .macbert_base，再执行：

    $env:PYTHONPATH = (Resolve-Path .bert_dependencies).Path
    & "$env:USERPROFILE\anaconda3\envs\llm\python.exe" train_macbert.py

[原模型及校验信息](https://huggingface.co/hfl/chinese-macbert-base/blob/a986e004d2a7f2a1c2f5a3edef4e20604a974ed1/pytorch_model.bin)

## 人工复核

将 reports/macbert/人工复核 下两份 CSV 分别交给两位标注员。每份均为同样的 300 条评论，只有正文和空白标注栏。样本覆盖低星、三星、高星及常见转折表达候选，已过滤常见显式星级写法。标注员按标注说明独立判断。对照键在完成第一轮独立标注后才打开。

填完后可统计一致率和分歧。指定两份已填写的 CSV，末尾可另加 --adjudicated 和独立复核后的完整标签表。三分类指标仅在人工可归入三类的样本上计算。

    $env:PYTHONPATH = (Resolve-Path .bert_dependencies).Path
    & "$env:USERPROFILE\anaconda3\envs\llm\python.exe" review_macbert_audit.py "已填1.csv" "已填2.csv"
