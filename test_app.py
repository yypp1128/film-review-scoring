from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parent
pytestmark=pytest.mark.skipif(not (ROOT/'models/baseline.joblib').exists(),reason='需要先训练模型')

def test_single_prediction_and_empty_input():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric)>=6
    at.text_area[0].set_value('').run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.warning)==1

def test_batch_testset_dashboard():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    at.checkbox[0].check().run()
    at.button[1].click().run()
    assert not at.exception
    assert 'batch_result' in at.session_state
    pred,overview,stats=at.session_state['batch_result']
    assert len(pred)>0 and len(overview)>0
    assert overview.score_10.between(2,10).all()
