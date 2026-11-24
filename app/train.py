import argparse
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_absolute_error
from core import clean_frame, split_groups, make_model, predict_frame, aggregate

ROOT=Path(__file__).resolve().parent

def metrics(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'per_class':classification_report(y,pred,labels=[0,1,2],output_dict=True,zero_division=0)}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,default=ROOT/'data/reviews.csv')
    args=parser.parse_args()
    out=ROOT/'reports'; out.mkdir(exist_ok=True)
    models=ROOT/'models'; models.mkdir(exist_ok=True)
    df,cleaning=clean_frame(pd.read_csv(args.data),require_labels=True)
    split=split_groups(df)
    df['split']=''
    for name,ix in split.items(): df.loc[ix,'split']=name
    df.to_csv(ROOT/'data/cleaned_with_split.csv',index=False,encoding='utf-8-sig')
    tr=df.iloc[split['train']]; va=df.iloc[split['validation']]; te=df.iloc[split['test']]
    scores={}; candidates=[]
    for kind,c in [('nb',1),('lr',.5),('lr',2)]:
        name=f'{kind}_C{c}'
        model=make_model(kind,c); model.fit(tr.text,tr.sentiment)
        val=metrics(va.sentiment,model.predict(va.text))
        scores[name]={'validation':val}
        candidates.append((val['macro_f1'],name,model))
        print(name,'validation Macro-F1',round(val['macro_f1'],4),flush=True)
    _,best_name,sentiment_model=max(candidates,key=lambda x:x[0])
    star_candidates=[]
    for c in [.5,2]:
        for weight in [None,'balanced']:
            model=make_model('lr',c,class_weight=weight); model.fit(tr.text,tr.star)
            expected=model.predict_proba(va.text)@model.classes_.astype(float)
            star_candidates.append((float(mean_absolute_error(va.star,expected)),c,weight,model))
    val_mae,star_c,star_weight,star_model=min(star_candidates,key=lambda x:x[0])
    bundle={'sentiment_model':sentiment_model,'star_model':star_model,
            'prior_mean':float(tr.star.mean()),'prior_strength':20,'selected_sentiment':best_name,
            'selected_star_c':star_c,'selected_star_weight':star_weight,'python':platform.python_version(),'sklearn':sklearn.__version__,
            'data_kind':'真实公开影评，学术研究用途','labels':'星级生成的弱情感标签'}
    # Models remain fitted on train only; validation is used exclusively for selection.
    for _,name,model in candidates:
        scores[name]['test']=metrics(te.sentiment,model.predict(te.text))
    majority=int(tr.sentiment.mode().iloc[0])
    scores['majority']={'test':metrics(te.sentiment,np.full(len(te),majority))}
    predictions,_=predict_frame(bundle,te)
    predicted=predictions.predicted_star.to_numpy()
    movie=aggregate(predictions,bundle['prior_mean'],20)
    actual=te.groupby('movie_id').star.mean()
    movie['actual_mean_star']=movie.movie_id.map(actual)
    star_results={'selected_c':star_c,'selected_class_weight':star_weight,'validation_mae':val_mae,
                  'validation_candidates':[{'mae':a,'c':b,'class_weight':w} for a,b,w,_ in star_candidates],
                  'test_mae':float(mean_absolute_error(te.star,predicted)),
                  'mean_baseline_mae':float(mean_absolute_error(te.star,np.full(len(te),tr.star.mean()))),
                  'movie_raw_mae':float(mean_absolute_error(movie.actual_mean_star,movie.mean_star)),
                  'movie_smoothed_mae':float(mean_absolute_error(movie.actual_mean_star,movie.smoothed_star)),
                  'test_movie_count':len(movie)}
    predictions['true_sentiment']=predictions.sentiment.map({0:'负向',1:'中性',2:'正向'})
    errors=predictions[predictions.sentiment_label!=predictions.true_sentiment]
    errors.head(100).to_csv(out/'error_examples.csv',index=False,encoding='utf-8-sig')
    predictions.to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig')
    movie.to_csv(out/'test_movies.csv',index=False,encoding='utf-8-sig')
    summary={'cleaning':cleaning,'split':{k:{'rows':len(ix),'movies':df.iloc[ix].movie_id.nunique(),
              'star_counts':{str(k):int(v) for k,v in df.iloc[ix].star.value_counts().items()}}
              for k,ix in split.items()},'selected_sentiment':best_name,'sentiment':scores,
              'star':star_results,'seed':42,'near_duplicate_removal':False,
              'limitations':['仅精确归一化去重，尚未清理语义近重复',
                            '情感标签由星级推导，尚未完成独立人工复核',
                            '固定单次按作品划分，尚未做多次交叉验证',
                            '开发期间曾查看固定测试结果，正式结论还需新增未查看作品复验',
                            '样本来自公开整理数据，不能代表平台全部用户']}
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    joblib.dump(bundle,models/'baseline.joblib')
    report=['# 基础实验运行结果','',
            f'清洗后 {len(df)} 条真实影评，{df.movie_id.nunique()} 部作品。按作品分组，随机种子 42。',
            '模型输入仅含评论正文，未使用原始星级、发布者预计算情感分数或作品名作为特征。','',
            '| 情感模型 | 验证 Macro-F1 | 测试 Macro-F1 | 测试准确率 |',
            '| --- | --- | --- | --- |']
    for name,item in scores.items():
        v=item.get('validation',{}).get('macro_f1')
        report.append(f"| {name} | {v if v is not None else '—'} | {item['test']['macro_f1']:.4f} | {item['test']['accuracy']:.4f} |")
    report += ['',f'依据验证集选择情感模型：{best_name}。测试集不参与模型选择。',
               f"五星预测测试 MAE：{star_results['test_mae']:.4f} 星；训练均值基线 MAE：{star_results['mean_baseline_mae']:.4f} 星。",
               f"测试作品原始均分 MAE：{star_results['movie_raw_mae']:.4f} 星；平滑均分 MAE：{star_results['movie_smoothed_mae']:.4f} 星。",
               '', '## 当前局限','']+['- '+x for x in summary['limitations']]
    report += ['', '已实现单页平台采集适配并测试静态解析，尚未进行线上采集实测。BERT 与独立人工标注尚未完成。本报告为实际运行结果，不代表工程实践全部要求均已完成。']
    (out/'基础实验结果.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'cleaning':cleaning,'selected':best_name,'star':star_results},ensure_ascii=True),flush=True)

if __name__=='__main__': main()
