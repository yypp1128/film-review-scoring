import numpy as np
import pandas as pd
import pytest
from core import clean_frame,split_groups,aggregate,predict_frame

def test_cleaning_keeps_negation_and_rejects_bad_labels():
    df=pd.DataFrame({'text':['不好看',' 不好看 ',None,'<b>还行吧</b>','很好看啊'],
                     'star':[1,1,4,3,9]})
    result,stats=clean_frame(df,True)
    assert result.text.tolist()==['不好看','还行吧']
    assert result.sentiment.tolist()==[0,1]
    assert stats['duplicate_rows']==1 and stats['invalid_labels']==1

def test_disjoint_movie_split():
    df=pd.DataFrame([{'movie_id':str(i),'star':s} for i in range(30) for s in range(1,6)])
    split=split_groups(df)
    groups=[set(df.iloc[v].movie_id) for v in split.values()]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert sum(map(len,split.values()))==len(df)

def test_aggregate_formula():
    df=pd.DataFrame({'movie_id':['a','a'],'movie_name':['电影']*2,
                     'predicted_star':[1,5],'sentiment_label':['负向','正向']})
    result=aggregate(df,prior_mean=4,prior_strength=2).iloc[0]
    assert result.mean_star==3 and result.smoothed_star==3.5 and result.score_10==7

class FakeModel:
    def __init__(self,classes,probs): self.classes_=np.array(classes); self.probs=probs
    def predict_proba(self,text): return np.tile(self.probs,(len(text),1))

def test_score_uses_class_values_and_not_column_positions():
    bundle={'sentiment_model':FakeModel([0,1,2],[.1,.2,.7]),
            'star_model':FakeModel([1,2,3,4,5],[.1,.1,.1,.2,.5])}
    result,_=predict_frame(bundle,pd.DataFrame({'text':['值得一看']}))
    assert result.iloc[0].predicted_star==pytest.approx(3.9)
    assert result.iloc[0].sentiment_label=='正向'
    with pytest.raises(ValueError): predict_frame(bundle,pd.DataFrame({'text':['']}))

def test_missing_text_column():
    with pytest.raises(ValueError): clean_frame(pd.DataFrame({'wrong':[1]}))

def test_collector_parser_on_fixture():
    from collect_page import parse_comments
    markup='<div class="comment-item"><span class="allstar40 rating"></span><span class="short">值得一看</span></div>'
    rows=parse_comments(markup,'123','样例')
    assert rows[0]['star']==4 and rows[0]['text']=='值得一看'
