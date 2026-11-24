# 版本记录

## v1.0.0 TF-IDF 增强版

网页完成单条影评评分、CSV 批量分析、作品汇总、下载及实验结果展示。评分使用字符级 TF-IDF 与逻辑回归。

[完整源码快照](https://github.com/yypp1128/film-review-scoring/tree/v1.0.0) · [实测报告](../app/reports/enhanced/增强版实测报告.md)

## v2.0.0 MacBERT 网页版

MacBERT 成为网页默认模型。可用同一页面、同一样本切换到 TF-IDF。新增模型训练、推理、人工复核工具和同划分对照报告。

[完整源码快照](https://github.com/yypp1128/film-review-scoring/tree/v2.0.0) · [实测报告](../app/reports/macbert/MacBERT对照实测报告.md)

| 40 部新增作品 4,749 条评论 | v1 TF-IDF | v2 MacBERT |
| --- | ---: | ---: |
| 情感 Macro-F1 | 0.474 | 0.506 |
| 负向召回率 | 31.9% | 34.7% |
| 中性召回率 | 40.8% | 38.2% |
| 单条星级 MAE | 0.684 | 0.661 |
| 作品均分 MAE | 0.235 | 0.219 |

MAE 以星为单位。三类真值由星级推导，人工盲标复核另行开展。

[比较两个版本的代码](https://github.com/yypp1128/film-review-scoring/compare/v1.0.0...v2.0.0)
