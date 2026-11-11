from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

def test_dashboard_pages_and_prediction():
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=40).run();assert not at.exception
    for page in ["Data explorer","Model evaluation","Feature experiments","Prediction lab","Methodology"]:
        at.sidebar.radio[0].set_value(page).run();assert not at.exception
        if page=="Prediction lab":
            at.button[0].click().run()
            assert not at.exception and at.metric
            assert any(metric.label=="Best validated model" for metric in at.metric)
