
"""Score a review or a UTF-8 CSV with the local MacBERT experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import load_file
import torch
from transformers import BertConfig, BertTokenizerFast

from core import SENTIMENTS, aggregate, clean_frame
from train_macbert import MacBertRating, STARS, encode, probabilities

ROOT = Path(__file__).resolve().parent


def load_model(model_dir, device):
    params = json.loads((model_dir/'scoring.json').read_text(encoding='utf-8'))
    tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = MacBertRating(BertConfig.from_pretrained(model_dir, local_files_only=True))
    model.load_state_dict(load_file(model_dir/'ratings.safetensors'))
    model.to(device).eval()
    dtype = torch.float32
    if device.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return model, tokenizer, params, device, dtype


def predict_loaded(frame, runtime):
    frame, stats = clean_frame(frame)
    if frame.empty:
        raise ValueError('清洗后没有有效评论，请提供 2 至 5000 字的评论。')
    model, tokenizer, params, device, dtype = runtime
    sentiment, stars = probabilities(model, tokenizer, encode(tokenizer, frame, params['max_length']),
                                     32, device, dtype)
    weighted = sentiment * np.exp(params['offsets'])
    weighted /= weighted.sum(axis=1, keepdims=True)
    frame['sentiment_label'] = [SENTIMENTS[i] for i in weighted.argmax(axis=1)]
    frame['confidence'] = weighted.max(axis=1)
    frame['needs_review'] = frame.confidence < .50
    frame['predicted_star'] = np.clip(4 + params['slope'] * (stars @ STARS - 4) + params['bias'], 1, 5)
    for i in range(3):
        frame[f'p_{i}'] = weighted[:, i]
    movies = aggregate(frame, params['prior_mean'], params['prior_strength'])
    return frame, movies, stats


def predict(frame, model_dir, device):
    return predict_loaded(frame, load_model(model_dir, device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text')
    inputs.add_argument('--csv', type=Path)
    inputs.add_argument('--interactive', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'reports/macbert/score_output')
    parser.add_argument('--model', type=Path, default=ROOT/'models/macbert')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.interactive:
        args.text = input('请输入一条中文影评，然后按回车：\n').strip()
    frame = (pd.DataFrame({'text':[args.text]}) if args.text is not None
             else pd.read_csv(args.csv, encoding='utf-8-sig'))
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    pred, movies, stats = predict(frame, args.model, device)
    if args.text is not None:
        row = pred.iloc[0]
        print(json.dumps({'情感':row.sentiment_label, '预测星级':round(row.predicted_star, 2),
                          '需复核':bool(row.needs_review)}, ensure_ascii=False))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        pred.to_csv(args.out/'review_predictions.csv', index=False, encoding='utf-8-sig')
        movies.to_csv(args.out/'movie_scores.csv', index=False, encoding='utf-8-sig')
        print(f'{stats["retained_rows"]} 条评论已评分，结果保存在 {args.out}')


if __name__ == '__main__':
    main()
