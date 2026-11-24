
"""Report annotator agreement and evaluate both models on adjudicated labels."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parent
LABELS = ['负向', '中性', '正向']
CHOICES = [*LABELS, '褒贬混合', '无法判断']
ID = '样本编号'
LABEL = '情感标签'


def load_labels(path, expected_ids):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    labels = frame.set_index(ID, verify_integrity=True)[LABEL].str.strip()
    if set(expected_ids) != set(labels.index):
        raise ValueError(f'{path.name} 的样本编号与清单不一致。')
    if not labels.isin(CHOICES).all():
        raise ValueError(f'{path.name} 含空白或不在说明中的情感标签。')
    return labels.reindex(expected_ids)


def human_metrics(accepted, file):
    predictions = pd.read_csv(file, encoding='utf-8-sig',
                              usecols=['movie_id', 'text', 'sentiment_label'])
    rows = accepted.merge(predictions, on=['movie_id', 'text'], validate='one_to_one')
    if len(rows) != len(accepted):
        raise ValueError(f'{file.name} 无法匹配全部人工样本。')
    return {
        'macro_f1': f1_score(rows[LABEL], rows.sentiment_label, labels=LABELS,
                             average='macro', zero_division=0),
        'negative_recall': recall_score(rows[LABEL], rows.sentiment_label, labels=['负向'],
                                        average='macro', zero_division=0),
        'rows':len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotator_one', type=Path)
    parser.add_argument('annotator_two', type=Path)
    parser.add_argument('--adjudicated', type=Path, help='已复核的完整 300 行表格')
    args = parser.parse_args()
    out = ROOT/'reports/macbert'
    template = pd.read_csv(out/'人工复核/标注员1.csv', encoding='utf-8-sig',
                           dtype=str, keep_default_na=False)
    ids = template[ID].tolist()
    first = load_labels(args.annotator_one, ids)
    second = load_labels(args.annotator_two, ids)
    agrees = (first == second)
    summary = {'annotated':len(ids), 'agreement':float(agrees.mean()),
               'cohen_kappa':float(cohen_kappa_score(first, second, labels=CHOICES))}
    reconciliation = template.copy()
    reconciliation['标注员1'] = first.to_numpy()
    reconciliation['标注员2'] = second.to_numpy()
    reconciliation[LABEL] = first.where(agrees, '').to_numpy()
    reconciliation.to_csv(out/'人工复核一致与分歧.csv', encoding='utf-8-sig', index=False)
    if args.adjudicated is not None or agrees.all():
        labels = load_labels(args.adjudicated, ids) if args.adjudicated else first
        summary['adjudicated_counts'] = labels.value_counts().to_dict()
        key = pd.read_csv(out/'人工复核对照键_标注完成后查看.csv',
                          encoding='utf-8-sig').rename(columns={'id':ID})
        accepted = template[[ID,'影评正文']].rename(columns={'影评正文':'text'}).merge(
            key[[ID,'movie_id']], on=ID, validate='one_to_one')
        accepted[LABEL] = labels.reindex(accepted[ID]).to_numpy()
        accepted = accepted[accepted[LABEL].isin(LABELS)]
        if not accepted.empty:
            summary['on_manual_labels'] = {
                'tfidf':human_metrics(accepted, ROOT/'reports/enhanced/test_predictions.csv'),
                'macbert':human_metrics(accepted, out/'test_predictions.csv'),
            }
    output = out/'人工复核统计.json'
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
