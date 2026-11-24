
"""Compare a joint MacBERT model with the existing TF-IDF system."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import mean_absolute_error
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

from core import SENTIMENTS, aggregate
from train import metrics
from train_enhanced import STARS, make_partitions, score_options, sentiment_options

ROOT = Path(__file__).resolve().parent
SHA256 = 'db0506d985574b80c33eec1cf13bd4c130585568753175871095ca13dbad9e23'
MODEL = 'hfl/chinese-macbert-base@a986e004d2a7f2a1c2f5a3edef4e20604a974ed1'
SEED = 42


class MacBertRating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(.1)
        self.sentiment = nn.Linear(config.hidden_size, 3)
        self.stars = nn.Linear(config.hidden_size, 5)

    def forward(self, batch):
        cls = self.dropout(self.encoder(**batch).last_hidden_state[:, 0])
        return self.sentiment(cls), self.stars(cls)


class Probabilities:
    def __init__(self, values):
        self.values = values

    def probabilities(self, _):
        return self.values


def load_pretrained(model, base):
    source = base/'pytorch_model.bin'
    with source.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != SHA256:
            raise ValueError('预训练权重与发布者公布的 SHA256 不一致。')
    state = torch.load(source, map_location='cpu', weights_only=True)
    encoder_state = {}
    for key, value in state.items():
        if key.startswith('bert.') and not key.startswith('bert.pooler.'):
            key = key.removeprefix('bert.').replace('.gamma', '.weight').replace('.beta', '.bias')
            if key in model.encoder.state_dict():
                encoder_state[key] = value
    model.encoder.load_state_dict(encoder_state, strict=True)


def encode(tokenizer, frame, max_length):
    return tokenizer(frame.text.tolist(), truncation=True, max_length=max_length,
                     return_token_type_ids=False)


def batches(enc, size, rng=None):
    lengths = np.asarray([len(ids) for ids in enc['input_ids']])
    order = np.arange(len(lengths)) if rng is None else rng.permutation(len(lengths))
    result = []
    for block in np.array_split(order, max(1, int(np.ceil(len(order)/(size*30))))):
        sorted_block = block[np.argsort(lengths[block], kind='stable')]
        result.extend(sorted_block[i:i+size] for i in range(0, len(block), size))
    if rng is not None:
        rng.shuffle(result)
    return result


def collate(tokenizer, enc, indices, device):
    sample = [{'input_ids': enc['input_ids'][i]} for i in indices]
    padded = tokenizer.pad(sample, padding=True, pad_to_multiple_of=8, return_tensors='pt')
    return {k:v.to(device) for k,v in padded.items()}


@torch.inference_mode()
def probabilities(model, tokenizer, enc, batch_size, device, dtype):
    model.eval()
    results = [np.zeros((len(enc['input_ids']), n)) for n in (3,5)]
    for ix in batches(enc, batch_size):
        with torch.autocast(device.type, dtype=dtype, enabled=device.type=='cuda'):
            logits = model(collate(tokenizer, enc, ix, device))
        for output, scores in zip(results, logits):
            output[ix] = scores.float().softmax(dim=1).cpu().numpy()
    return results


def annotate(partitions, out):
    test = partitions['test']
    # Keep the hand-labeled audit away from training, validation and threshold selection.
    explicit = test.text.str.contains(r'(?:[一二三四五1-5]\s*[颗个]?\s*星|[0-9]+(?:\.[0-9]+)?\s*分|[★☆⭐])')
    pool = test.loc[~explicit]
    contrast = pool.text.str.contains(r'但是|不过|可惜|然而|虽然|尽管|遗憾|唯一不足|美中不足')
    groups = []
    for i, (label, group) in enumerate([
        ('contrast', pool[contrast]),
        ('low', pool[~contrast & (pool.star <= 2)]),
        ('three', pool[~contrast & (pool.star == 3)]),
        ('high', pool[~contrast & (pool.star >= 4)]),
    ]):
        picked = group.sample(n=75, random_state=SEED+i).copy()
        picked['sample_stratum'] = label
        groups.append(picked)
    assigned = pd.concat(groups).sample(frac=1, random_state=SEED).reset_index(drop=True)
    assigned.insert(0, 'id', [f'RV{i:04}' for i in range(1, len(assigned)+1)])
    blind = assigned[['id','text']].rename(columns={'id':'样本编号', 'text':'影评正文'})
    for field in ('情感标签','备注'):
        blind[field] = ''
    folder = out/'人工复核'
    folder.mkdir(exist_ok=True)
    for i in (1,2):
        blind.to_csv(folder/f'标注员{i}.csv', index=False, encoding='utf-8-sig')
    assigned[['id','movie_id','star','sentiment','sample_stratum']].to_csv(
        out/'人工复核对照键_标注完成后查看.csv', index=False, encoding='utf-8-sig')
    (folder/'标注说明.txt').write_text(
        '两名标注员各使用一份表，只根据正文独立填“情感标签”。\n'
        '可填：正向、中性、负向、褒贬混合、无法判断。无明确态度填中性，'
        '同时存在鲜明的正反态度且难以判断主导方向时填褒贬混合。\n'
        '反讽按真实语气判断。不确定时在备注写下依据。请勿互看标签或查看对照键。\n'
        '两人填完后按样本编号比对。记录初始一致率，再讨论分歧并保存一致标签。'
        '混合与无法判断单独报告。三类指标只在可归入三类的人工样本上计算。\n'
        '这是特意均衡抽样的困难样本，比例不代表全量测试集。正文中的显式星级只按常见写法过滤。\n',
        encoding='utf-8-sig')


def check_splits(partitions, baseline):
    for name, frame in partitions.items():
        expected = baseline['partitions'][name]
        if sorted(frame.movie_id.unique()) != expected['movie_ids'] or len(frame) != expected['rows']:
            raise ValueError(f'现有模型与 MacBERT 的 {name} 划分不一致。')


def evaluate(frame, prob, params, prior):
    sentiment, stars = prob
    weighted = sentiment * np.exp(params['offsets'])
    weighted /= weighted.sum(axis=1, keepdims=True)
    labels = weighted.argmax(axis=1)
    result = frame[['movie_id','movie_name','text','star','sentiment']].copy()
    result['sentiment_label'] = [SENTIMENTS[i] for i in labels]
    result['confidence'] = weighted.max(axis=1)
    result['needs_review'] = result.confidence < .5
    result['predicted_star'] = np.clip(4 + params['slope'] * (stars @ STARS - 4) + params['bias'], 1, 5)
    for i in range(3):
        result[f'p_{i}'] = weighted[:,i]
    movies = aggregate(result, prior, 20)
    movies['actual_mean_star'] = movies.movie_id.map(frame.groupby('movie_id').star.mean())
    return {
        'sentiment': metrics(frame.sentiment, labels),
        'star_mae': mean_absolute_error(frame.star, result.predicted_star),
        'movie_mae': mean_absolute_error(movies.actual_mean_star, movies.mean_star),
        'movie_smoothed_mae': mean_absolute_error(movies.actual_mean_star, movies.smoothed_star),
    }, result, movies


def write_report(out, report):
    lines = ['# MacBERT 与原模型的同划分对照', '',
             '沿用增强版固定训练集及按作品隔离的验证集、新测试集和旧测试集。模型输入只有影评正文。'
             '以验证集选择检查点及调整参数，两个测试集仅用于报告。', '',
             '| 测试集 | 指标 | TF-IDF 增强版 | MacBERT |', '| --- | --- | ---: | ---: |']
    fields = [
        ('情感 Macro-F1', lambda s:s['sentiment']['macro_f1']),
        ('负向召回率', lambda s:s['sentiment']['per_class']['0']['recall']),
        ('中性召回率', lambda s:s['sentiment']['per_class']['1']['recall']),
        ('准确率', lambda s:s['sentiment']['accuracy']),
        ('单条五星 MAE', lambda s:s['star_mae']),
        ('作品原均分 MAE', lambda s:s['movie_mae']),
    ]
    for key, label in [('test','新增作品'),('legacy','原测试集')]:
        for title, fn in fields:
            a,b = (fn(report['evaluation'][key][name]) for name in ('tfidf','macbert'))
            lines.append(f'| {label} | {title} | {a:.4f} | {b:.4f} |')
    lines.extend(['', 'MAE 单位为星，作品平均按作品等权计。三类情感真值来自星级映射。'
                  '人工盲标复核尚无标注结果，不能由这些数值推断人工标签上的表现。', '',
                  f"选中第 {report['selected']['epoch']} 轮。每轮验证情况见 metrics.json。"])
    (out/'MacBERT对照实测报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'.macbert_base')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=96)
    parser.add_argument('--freeze-layers', type=int, default=6)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    rng = np.random.default_rng(SEED)
    out = ROOT/'reports/macbert'
    out.mkdir(parents=True, exist_ok=True)
    data = make_partitions()
    baseline = json.loads((ROOT/'reports/enhanced/metrics.json').read_text(encoding='utf-8'))
    check_splits(data, baseline)
    if not (out/'人工复核对照键_标注完成后查看.csv').exists():
        annotate(data, out)
    if args.audit_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('本实验需要可用的 CUDA GPU。')
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.set_float32_matmul_precision('high')
    train, val = data['train'], data['validation']
    config = BertConfig.from_json_file(str(args.base/'config.json'))
    tokenizer = BertTokenizerFast(vocab_file=str(args.base/'vocab.txt'))
    model = MacBertRating(config)
    load_pretrained(model, args.base)
    model.encoder.embeddings.requires_grad_(False)
    for layer in model.encoder.encoder.layer[:args.freeze_layers]:
        layer.requires_grad_(False)
    model.to(device)
    print(f'GPU: {torch.cuda.get_device_name()}, dtype={dtype}', flush=True)
    optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),
                                  lr=3e-5, weight_decay=.01, fused=True)
    y_sentiment = torch.tensor(train.sentiment.to_numpy(), device=device, dtype=torch.long)
    y_stars = torch.tensor(train.star.to_numpy()-1, device=device, dtype=torch.long)
    encoded = {key:encode(tokenizer, frame, args.max_length) for key,frame in data.items()}
    train_batches = [batches(encoded['train'], args.batch_size, rng) for _ in range(args.epochs)]
    total_steps = sum(map(len, train_batches))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.08*total_steps), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype==torch.float16)
    loss_fn = nn.CrossEntropyLoss()
    best_value = -np.inf
    model_dir = ROOT/'models/macbert'
    model_dir.mkdir(exist_ok=True)
    config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    checkpoint = model_dir/'ratings.safetensors'
    history = []
    for epoch, indices in enumerate(train_batches, 1):
        model.train()
        for step, ix in enumerate(indices, 1):
            optimizer.zero_grad(set_to_none=True)
            ix_gpu = torch.as_tensor(ix, device=device)
            with torch.autocast('cuda', dtype=dtype):
                sent, stars = model(collate(tokenizer, encoded['train'], ix, device))
                loss = .5*loss_fn(sent, y_sentiment[ix_gpu]) + loss_fn(stars, y_stars[ix_gpu])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if step == 1 or step % 50 == 0:
                print(f'epoch {epoch}, step {step}/{len(indices)}, loss {loss.item():.4f}', flush=True)
        sent, stars = probabilities(model, tokenizer, encoded['validation'], args.batch_size, device, dtype)
        f1, _, offsets = sentiment_options(None, val.sentiment,
                                           {'macbert':Probabilities(sent)})
        score, _, slope, bias = score_options(None, val, {'macbert':Probabilities(stars)})
        params = {'epoch':epoch, 'f1':f1, 'scoring_loss':score, 'offsets':offsets,
                  'slope':slope, 'bias':bias}
        history.append(params)
        print('validation', params, flush=True)
        if f1 - score > best_value:
            best_value = f1 - score
            best = params
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, checkpoint)
            (model_dir/'scoring.json').write_text(json.dumps({
                **best, 'max_length':args.max_length, 'prior_mean':float(train.star.mean()),
                'prior_strength':20, 'pretrained':MODEL,
            }, indent=2), encoding='utf-8')
    model.load_state_dict(load_file(checkpoint))
    report = {'pretrained':MODEL, 'sha256':SHA256, 'gpu':torch.cuda.get_device_name(),
              'precision':str(dtype), 'partitions':baseline['partitions'], 'selected':best,
              'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'seed':SEED, 'validation':history, 'evaluation':{}}
    for key in ('test', 'legacy'):
        outputs = probabilities(model, tokenizer, encoded[key], args.batch_size, device, dtype)
        result, preds, movies = evaluate(data[key], outputs, best, train.star.mean())
        preds.to_csv(out/f'{key}_predictions.csv', index=False, encoding='utf-8-sig')
        movies.to_csv(out/f'{key}_movies.csv', index=False, encoding='utf-8-sig')
        report['evaluation'][key] = {'tfidf':baseline['evaluation'][key]['enhanced'], 'macbert':result}
        print(key, result['sentiment']['macro_f1'], result['star_mae'], result['movie_mae'], flush=True)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding='utf-8')
    write_report(out, report)


if __name__ == '__main__':
    main()
