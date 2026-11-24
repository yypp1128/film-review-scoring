"""Train and compare scoring models on fixed, disjoint movie groups."""
import json
import hashlib
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error
from threadpoolctl import threadpool_limits

from core import aggregate, clean_frame, predict_frame, split_groups
from rating_model import TextModel
from train import metrics

ROOT = Path(__file__).resolve().parent
STARS = np.arange(1, 6)
SENTIMENTS = np.arange(3)


def review_key(frame):
    return frame.text.str.replace(r'\s+', '', regex=True)


def make_partitions():
    original, _ = clean_frame(pd.read_csv(ROOT/'data/reviews.csv'), require_labels=True)
    legacy = original.iloc[split_groups(original)['test']].copy()
    expanded = pd.read_csv(ROOT/'data/expanded_reviews.csv')
    old_ids = set(original.movie_id)
    legacy_ids = set(legacy.movie_id)
    new_ids = sorted(set(expanded.movie_id) - old_ids,
                     key=lambda key: hashlib.sha256(('evaluation-v1:' + key).encode()).digest())
    fresh_ids, validation_ids = set(new_ids[:40]), set(new_ids[40:80])
    sections = {}
    reserved = set()
    # Keep the historical comparison rows intact and exclude their text from all other sets.
    for name, rows in [
        ('legacy', legacy),
        ('test', expanded[expanded.movie_id.isin(fresh_ids)]),
        ('validation', expanded[expanded.movie_id.isin(validation_ids)]),
        ('train', expanded[~expanded.movie_id.isin(legacy_ids | fresh_ids | validation_ids)]),
    ]:
        clean, _ = clean_frame(rows, require_labels=True)
        keys = review_key(clean)
        sections[name] = clean.loc[~keys.isin(reserved)].reset_index(drop=True)
        reserved.update(keys)
    for name, frame in sections.items():
        if set(frame.star) != set(STARS):
            raise ValueError(f'{name} must cover all five stars')
        print(name, len(frame), frame.movie_id.nunique(),
              frame.star.value_counts().sort_index().to_dict(), flush=True)
    return sections


def sentiment_options(matrix, labels, candidates):
    results = []
    for name, model in candidates.items():
        p = model.probabilities(matrix)
        log_p = np.log(np.maximum(p, 1e-12))
        for negative, neutral in product(np.arange(0, 2.01, .25), repeat=2):
            offsets = [negative, neutral, 0]
            predictions = (log_p + offsets).argmax(axis=1)
            score = f1_score(labels, predictions, average='macro')
            results.append((score, name, offsets))
    return max(results, key=lambda row: row[0])


def score_options(matrix, validation, candidates):
    results = []
    groups = validation.movie_id
    actual = validation.star.groupby(groups).mean()
    for name, model in candidates.items():
        expected = model.probabilities(matrix) @ STARS
        for slope, bias in product([1., 1.25, 1.5, 1.75], np.arange(-.15, .251, .05)):
            preds = np.clip(4 + slope * (expected - 4) + bias, 1, 5)
            individual = mean_absolute_error(validation.star, preds)
            group = mean_absolute_error(actual, pd.Series(preds).groupby(groups).mean())
            results.append((.8 * individual + .2 * group, name, float(slope), float(bias)))
    return min(results, key=lambda row: row[0])


def evaluate(bundle, frame):
    pred, _ = predict_frame(bundle, frame)
    pred['true_sentiment'] = pred.sentiment.map({0: '负向', 1: '中性', 2: '正向'})
    labels = pred.sentiment_label.map({'负向': 0, '中性': 1, '正向': 2})
    films = aggregate(pred, bundle['prior_mean'], bundle['prior_strength'])
    films['actual_mean_star'] = films.movie_id.map(pred.groupby('movie_id').star.mean())
    return {
        'sentiment': metrics(pred.sentiment, labels),
        'star_mae': mean_absolute_error(pred.star, pred.predicted_star),
        'movie_mae': mean_absolute_error(films.actual_mean_star, films.mean_star),
        'movie_smoothed_mae': mean_absolute_error(films.actual_mean_star, films.smoothed_star),
    }, pred, films


def main():
    partitions = make_partitions()
    train, validation = partitions['train'], partitions['validation']
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), min_df=3,
                                 max_features=120000, sublinear_tf=True, dtype=np.float64)
    x = vectorizer.fit_transform(train.text)
    v = vectorizer.transform(validation.text)
    print('Features:', x.shape, flush=True)

    star = {}
    sentiment = {}
    for c in [1., 4., 12.]:
        thresholds = [LogisticRegression(C=c, solver='liblinear', random_state=42).fit(x, train.star > t)
                      for t in range(1, 5)]
        ordinal = TextModel(vectorizer, thresholds, STARS, 'ordinal')
        nominal = LogisticRegression(C=c, max_iter=450, tol=0.0005).fit(x, train.star)
        five = TextModel(vectorizer, [nominal], STARS, 'nominal')
        star[f'ordinal_{c:g}'] = ordinal
        star[f'five_{c:g}'] = five
        star[f'blend_{c:g}'] = TextModel(vectorizer, [ordinal, five], STARS, 'blend')
        for name in [f'ordinal_{c:g}', f'five_{c:g}', f'blend_{c:g}']:
            sentiment[name] = TextModel(vectorizer, [star[name]], SENTIMENTS, 'collapsed')
        classifier = LogisticRegression(C=c, max_iter=450, tol=0.0005).fit(x, train.sentiment)
        sentiment[f'three_{c:g}'] = TextModel(vectorizer, [classifier], SENTIMENTS, 'nominal')
        print(f'Trained candidates C={c:g}', flush=True)

    validation_f1, sentiment_name, offsets = sentiment_options(v, validation.sentiment, sentiment)
    validation_loss, star_name, slope, bias = score_options(v, validation, star)
    sentiment_model = sentiment[sentiment_name]
    sentiment_model.offsets = offsets
    bundle = {
        'sentiment_model': sentiment_model, 'star_model': star[star_name],
        'star_slope': slope, 'star_bias': bias, 'prior_mean': float(train.star.mean()),
        'prior_strength': 20, 'selected_sentiment': sentiment_name, 'selected_star': star_name,
    }
    print('Selected:', sentiment_name, offsets, validation_f1, star_name, slope, bias,
          validation_loss, flush=True)
    baseline = joblib.load(ROOT/'models/baseline.joblib')
    reports = ROOT/'reports/enhanced'
    reports.mkdir(exist_ok=True)
    result = {
        'selected_sentiment': sentiment_name, 'selected_star': star_name,
        'sentiment_offsets': offsets, 'star_slope': slope, 'star_bias': bias,
        'validation_f1': validation_f1, 'validation_scoring_loss': validation_loss,
        'partitions': {name: {'rows': len(df), 'movies': int(df.movie_id.nunique()),
                              'stars': {str(k): int(v) for k, v in df.star.value_counts().sort_index().items()},
                              'movie_ids': sorted(df.movie_id.unique())}
                       for name, df in partitions.items()},
        'evaluation': {},
    }
    for name in ('legacy', 'test'):
        frame = partitions[name]
        old, _, _ = evaluate(baseline, frame)
        new, predictions, films = evaluate(bundle, frame)
        result['evaluation'][name] = {'baseline': old, 'enhanced': new}
        predictions.to_csv(reports/f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
        films.to_csv(reports/f'{name}_movies.csv', index=False, encoding='utf-8-sig')
        for variant, score in result['evaluation'][name].items():
            print(name, variant, 'F1', score['sentiment']['macro_f1'],
                  'MAE', score['star_mae'], 'movie MAE', score['movie_mae'], flush=True)
    (reports/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    joblib.dump(bundle, ROOT/'models/enhanced.joblib', compress=3)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
