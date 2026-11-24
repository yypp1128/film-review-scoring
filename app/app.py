from pathlib import Path
import json
import io
from importlib.util import find_spec
from threading import Lock
import joblib
import pandas as pd
import streamlit as st
from core import predict_frame, aggregate

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='影评实验室',page_icon='🎬',layout='wide')
st.markdown('''<style>
.stApp { background:#f3f0e9; color:#202c29; }
h1,h2,h3 {font-family:"Noto Serif SC","SimSun",serif;letter-spacing:.04em;}
div[data-testid="stMetric"] {background:#e3e8df;padding:18px;border-top:3px solid #386756;}
.stButton>button {background:#235545;color:white;border:0;}
</style>''',unsafe_allow_html=True)
st.caption('FILM / LANGUAGE / EVIDENCE')
st.title('影评实验室')
st.write('从文字理解态度，以真实实验检验评分。')
enhanced_path=ROOT/'models/enhanced.joblib'
enhanced=enhanced_path.is_file()
model_path=enhanced_path if enhanced else ROOT/'models/baseline.joblib'
macbert_path=ROOT/'models/macbert'
macbert_installed=all((macbert_path/name).is_file() for name in (
    'ratings.safetensors', 'scoring.json', 'config.json', 'tokenizer.json'))
macbert_available=macbert_installed and all(find_spec(name) is not None for name in (
    'torch', 'transformers', 'safetensors'))
models={}
if macbert_available:
    models['MacBERT']= 'macbert'
if model_path.is_file():
    models['TF-IDF 增强版' if enhanced else 'TF-IDF 基础版']='tfidf'
if not models:
    st.info('尚未训练模型。请先运行“一键准备实验.cmd”，完成数据下载和训练后刷新。')
    st.stop()

@st.cache_resource
def load_bundle(path, stamp):
    return joblib.load(path)

@st.cache_resource(show_spinner='正在加载本地 MacBERT 模型…')
def load_macbert(path, stamps):
    import torch
    from score_macbert import load_model

    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return load_model(path, device), Lock()


def run_scoring(frame):
    if selected == 'macbert':
        from score_macbert import predict_loaded

        stamps=tuple((macbert_path/name).stat().st_mtime_ns for name in (
            'ratings.safetensors', 'scoring.json', 'config.json', 'tokenizer.json'))
        runtime, lock=load_macbert(macbert_path, stamps)
        with lock:
            predictions, overview, stats=predict_loaded(frame, runtime)
        return predictions, overview, stats
    bundle=load_bundle(model_path, model_path.stat().st_mtime_ns)
    predictions, stats=predict_frame(bundle, frame)
    overview=aggregate(predictions, bundle['prior_mean'], bundle['prior_strength'])
    return predictions, overview, stats


model_name=st.selectbox('评分模型', list(models))
selected=models[model_name]
if macbert_installed and not macbert_available:
    st.caption('当前 Python 环境缺少 MacBERT 依赖。使用“启动系统.cmd”可加载 MacBERT。')
st.caption('三类情感标签由星级推导。评分反映样本口碑估计；十分值为预测星级乘以二，范围 2 至 10。')
single,batch,experiment=st.tabs(['01 单条影评','02 作品看板','03 实验结果'])

with single:
    text=st.text_area('输入中文影评',value='演员表演自然，故事很感人，值得一看。',height=150,max_chars=5000)
    if st.button('分析评论',type='primary'):
        try:
            pred,_,_=run_scoring(pd.DataFrame({'text':[text]}))
            row=pred.iloc[0]
            a,b,c=st.columns(3)
            a.metric('情感倾向',row.sentiment_label)
            b.metric('预测星级',f'{row.predicted_star:.2f} / 5')
            c.metric('最大类别概率',f'{row.confidence:.1%}')
            st.bar_chart(pd.DataFrame({'概率':[row.p_0,row.p_1,row.p_2]},index=['负向','中性','正向']))
            if row.needs_review: st.info('类别概率接近，建议人工复核。模型概率尚未进行独立校准。')
        except ValueError as exc: st.warning(str(exc))
        except (ImportError, OSError, RuntimeError) as exc:
            st.error(f'{model_name} 评分失败：{exc}')

with batch:
    st.write('上传含 text 列的 UTF-8 CSV，可附 movie_id 和 movie_name。未提供作品编号时按同一部作品汇总。')
    upload=st.file_uploader('评论文件',type=['csv'])
    use_test=st.checkbox('使用随附的真实测试集评论体验')
    if st.button('生成作品看板'):
        try:
            if upload is not None:
                if upload.size>20*1024*1024: raise ValueError('文件请控制在 20 MB 以内。')
                frame=pd.read_csv(io.BytesIO(upload.getvalue()),encoding='utf-8-sig')
            elif use_test:
                sample_path='reports/enhanced/test_predictions.csv' if enhanced else 'reports/test_predictions.csv'
                frame=pd.read_csv(ROOT/sample_path)[['movie_id', 'movie_name', 'text']]
            else:
                raise ValueError('请上传 CSV，或选择使用随附测试集。')
            if len(frame)>50000: raise ValueError('单次最多分析 50000 条评论。')
            with st.spinner(f'正在使用 {model_name} 分析评论…'):
                pred,overview,stats=run_scoring(frame)
            st.session_state[f'batch_result_{selected}']=(pred,overview,stats)
        except (ValueError,UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError) as exc:
            st.warning(f'文件处理失败：{exc}')
        except (ImportError, OSError, RuntimeError) as exc:
            st.error(f'{model_name} 评分失败：{exc}')
    if f'batch_result_{selected}' in st.session_state:
        pred,overview,stats=st.session_state[f'batch_result_{selected}']
        a,b,c=st.columns(3)
        a.metric('有效评论',len(pred)); b.metric('作品数',len(overview)); c.metric('过滤重复',stats['duplicate_rows'])
        st.subheader('作品口碑')
        display=overview.rename(columns={'movie_name':'作品','reviews':'评论数','mean_star':'预测均分',
                'smoothed_star':'平滑星级','score_10':'十分值','positive_ratio':'正向占比'})
        st.dataframe(display.drop(columns='movie_id'),use_container_width=True,hide_index=True)
        st.bar_chart(pred.sentiment_label.value_counts())
        st.caption('平滑星级结合训练集均值，先验强度固定为 20。重复评论只保留一次，样本多的作品有更充分的依据。')
        st.download_button('下载作品评分',overview.to_csv(index=False).encode('utf-8-sig'),'movie_scores.csv','text/csv')
        st.download_button('下载逐条预测',pred.to_csv(index=False).encode('utf-8-sig'),'review_predictions.csv','text/csv')
        st.dataframe(pred[['movie_name','text','sentiment_label','predicted_star','needs_review']],hide_index=True)

with experiment:
    path=ROOT/'reports/metrics.json'
    enhanced_report=ROOT/'reports/enhanced/metrics.json'
    macbert_report=ROOT/'reports/macbert/metrics.json'
    if macbert_report.is_file() and (selected == 'macbert' or enhanced):
        result=json.loads(macbert_report.read_text(encoding='utf-8'))
        fresh=result['evaluation']['test'][selected]
        score=fresh['sentiment']
        st.caption(f'当前模型 · {model_name}')
        a,b,c=st.columns(3)
        a.metric('新作品测试 Macro-F1',f"{score['macro_f1']:.3f}")
        b.metric('单条五星预测 MAE',f"{fresh['star_mae']:.3f}")
        c.metric('作品均分 MAE',f"{fresh['movie_mae']:.3f}")
        test=result['partitions']['test']
        st.write(f"新增测试集包含 {test['movies']} 部作品、{test['rows']:,} 条评论。两种模型使用相同的测试样本。")
        rows=[]
        for group, title in [('test', '新增作品'), ('legacy', '原测试集')]:
            for version, label in [('tfidf', 'TF-IDF 增强版'), ('macbert', 'MacBERT')]:
                item=result['evaluation'][group][version]
                sentiment=item['sentiment']
                rows.append({'测试集':title, '模型':label,
                             '情感 Macro-F1':round(sentiment['macro_f1'], 3),
                             '负向召回率':f"{sentiment['per_class']['0']['recall']:.1%}",
                             '中性召回率':f"{sentiment['per_class']['1']['recall']:.1%}",
                             '单条 MAE（星）':round(item['star_mae'], 3),
                             '作品均分 MAE（星）':round(item['movie_mae'], 3)})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.write('当前模型在新增作品上的混淆矩阵，行是真实类别，列是预测类别。')
        labels=['负向','中性','正向']
        st.dataframe(pd.DataFrame(score['confusion_matrix'],index=labels,columns=labels))
        st.caption('情感真值由原星级推导，人工盲标复核尚待完成。')
    elif enhanced and enhanced_report.is_file():
        result=json.loads(enhanced_report.read_text(encoding='utf-8'))
        fresh=result['evaluation']['test']['enhanced']
        score=fresh['sentiment']
        a,b,c=st.columns(3)
        a.metric('新作品测试 Macro-F1',f"{score['macro_f1']:.3f}")
        b.metric('单条五星预测 MAE',f"{fresh['star_mae']:.3f}")
        c.metric('作品均分 MAE',f"{fresh['movie_mae']:.3f}")
        st.write(f"训练 {result['partitions']['train']['rows']:,} 条评论。"
                 f"新增测试集包含 {result['partitions']['test']['movies']} 部作品，模型仅使用评论正文。")
        rows=[]
        for group, title in [('test', '新增作品'), ('legacy', '原测试集')]:
            for version, label in [('baseline', '基础版'), ('enhanced', '增强版')]:
                item=result['evaluation'][group][version]
                rows.append({'测试集': title, '模型': label,
                             '情感 Macro-F1': round(item['sentiment']['macro_f1'], 3),
                             '单条 MAE': round(item['star_mae'], 3),
                             '作品均分 MAE': round(item['movie_mae'], 3)})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.write('新增测试集混淆矩阵，行是真实类别，列是预测类别。')
        labels=['负向','中性','正向']
        st.dataframe(pd.DataFrame(score['confusion_matrix'],index=labels,columns=labels))
        st.caption('情感标签由星级推导，新增测试作品仍来自同一公开数据源。')
    elif path.exists():
        result=json.loads(path.read_text(encoding='utf-8'))
        selected=result['selected_sentiment']
        score=result['sentiment'][selected]['test']
        a,b,c=st.columns(3)
        a.metric('测试 Macro-F1',f"{score['macro_f1']:.3f}")
        b.metric('五星预测 MAE',f"{result['star']['test_mae']:.3f}")
        c.metric('清洗后真实评论',result['cleaning']['retained_rows'])
        st.write('训练、验证、测试集按作品分组；不同集合不包含相同作品。模型仅输入评论正文。')
        table=[]
        for name,item in result['sentiment'].items():
            table.append({'模型':name,'测试 Macro-F1':item['test']['macro_f1'],
                          '测试准确率':item['test']['accuracy'],
                          '验证 Macro-F1':item.get('validation',{}).get('macro_f1')})
        st.dataframe(pd.DataFrame(table),hide_index=True)
        st.write('混淆矩阵：行是真实类别，列是预测类别。')
        st.dataframe(pd.DataFrame(score['confusion_matrix'],index=['负向','中性','正向'],columns=['负向','中性','正向']))
        for limitation in result['limitations']: st.write('• '+limitation)
        st.caption('BERT、人工标签复核及平台直接采集尚未完成。')
    st.markdown('[数据发布说明](https://github.com/bbublm/Douban-Movie-Review-Dataset) · 仅供学术研究和个人学习')
