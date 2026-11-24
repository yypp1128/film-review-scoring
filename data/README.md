# 数据说明

语料取自固定版本的公开影评样本。同一部作品不会同时出现在训练、验证或测试集合中。训练只使用评论正文，原始星级用于两任务的训练标签和评估。

| 文件 | 内容 |
| --- | --- |
| [原始样本来源](../app/data/provenance.json) | 基础实验语料来源、固定提交和逐文件校验 |
| [扩充样本来源](../app/data/expanded_provenance.json) | 扩充数据的来源、固定提交和逐文件校验 |
| [固定作品划分](../app/reports/enhanced/metrics.json) | 增强版与 MacBERT 共用的划分信息，见 partitions |
| [数据准备代码](../app/prepare_data.py) | 下载、校验及整理为项目 CSV |

原始数据文件和逐条预测保存在本地交付包中。语料用于个人学习及学术研究。[数据发布说明](https://github.com/bbublm/Douban-Movie-Review-Dataset)
