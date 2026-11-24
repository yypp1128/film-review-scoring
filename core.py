"""Text-only baselines, leakage-aware splitting, prediction and aggregation."""
import html
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

SENTIMENTS = ['负向', '中性', '正向']

def normalize(text):
    if pd.isna(text):
        return ''
    text = html.unescape(unicodedata.normalize('NFKC', str(text)))
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', text)).strip()

def clean_frame(frame, require_labels=False):
    if 'text' not in frame:
        raise ValueError('文件必须包含 text 列。')
    df = frame.copy()
    stats = {'input_rows': len(df)}
    df['text'] = df['text'].map(normalize)
    df = df[df.text.str.len().between(2, 5000)].copy()
    stats['empty_or_invalid_length'] = stats['input_rows'] - len(df)
    df['_key'] = df.text.str.replace(r'\s+', '', regex=True)
    n = len(df)
    df = df.drop_duplicates('_key').drop(columns='_key').copy()
    stats['duplicate_rows'] = n - len(df)
    if 'movie_id' not in df:
        df['movie_id'] = 'uploaded'
    df['movie_id'] = df.movie_id.fillna('uploaded').astype(str)
    if 'movie_name' not in df:
        df['movie_name'] = df.movie_id
    df['movie_name'] = df.movie_name.fillna(df.movie_id).astype(str)
    if require_labels:
        if 'star' not in df:
            raise ValueError('训练数据必须包含 star 列，取值为整数 1 至 5。')
        df['star'] = pd.to_numeric(df.star, errors='coerce')
        n = len(df)
        df = df[df.star.isin([1, 2, 3, 4, 5])].copy()
        stats['invalid_labels'] = n - len(df)
        df['star'] = df.star.astype(int)
        df['sentiment'] = np.select([df.star <= 2, df.star == 3], [0, 1], default=2)
    stats['retained_rows'] = len(df)
    return df.reset_index(drop=True), stats

def split_groups(df, seed=42):
    if df.movie_id.nunique() < 10:
        raise ValueError('正式训练至少需要 10 部作品，以便按作品划分训练、验证和测试集。')
    first = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed)
    train, held = next(first.split(df, groups=df.movie_id))
    second = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed)
    val_local, test_local = next(second.split(df.iloc[held], groups=df.iloc[held].movie_id))
    result = {'train': train, 'validation': held[val_local], 'test': held[test_local]}
    for name, indices in result.items():
        if set(df.iloc[indices].star) != {1, 2, 3, 4, 5}:
            raise ValueError(f'{name} 未覆盖全部五星标签，请扩大数据，不要反复更换种子挑选测试集。')
    return result

def make_model(kind='lr', c=1, class_weight='balanced'):
    classifier = (LogisticRegression(C=c, class_weight=class_weight, max_iter=1500, random_state=42)
                  if kind == 'lr' else MultinomialNB(alpha=1.0))
    return Pipeline([('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4),
                       min_df=2, max_features=60000, sublinear_tf=True)), ('classifier', classifier)])

def predict_frame(bundle, frame):
    df, stats = clean_frame(frame)
    if df.empty:
        raise ValueError('清洗后没有有效评论，请提供 2 至 5000 字的评论。')
    sp = bundle['sentiment_model'].predict_proba(df.text)
    classes = bundle['sentiment_model'].classes_.astype(int)
    labels = classes[sp.argmax(axis=1)]
    star_model = bundle['star_model']
    rp = star_model.predict_proba(df.text)
    expected = rp @ star_model.classes_.astype(float)
    df['predicted_star'] = np.clip(4 + bundle.get('star_slope', 1) * (expected - 4)
                                   + bundle.get('star_bias', 0), 1, 5)
    df['sentiment_label'] = [SENTIMENTS[i] for i in labels]
    df['confidence'] = sp.max(axis=1)
    df['needs_review'] = df.confidence < .50
    for i, cls in enumerate(classes):
        df[f'p_{cls}'] = sp[:, i]
    return df, stats

def aggregate(predicted, prior_mean, prior_strength=20):
    if prior_strength < 0:
        raise ValueError('先验强度不能为负。')
    rows=[]
    for movie_id, group in predicted.groupby('movie_id', sort=False):
        n=len(group); avg=float(group.predicted_star.mean())
        smooth=(n*avg+prior_strength*prior_mean)/(n+prior_strength)
        rows.append({'movie_id':movie_id,'movie_name':group.movie_name.iloc[0],
                     'reviews':n,'mean_star':avg,'smoothed_star':smooth,'score_10':2*smooth,
                     'positive_ratio':float((group.sentiment_label=='正向').mean())})
    return pd.DataFrame(rows)
