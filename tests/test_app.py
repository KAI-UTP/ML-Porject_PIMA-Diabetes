from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Overview", "Data explorer", "Features", "Model evaluation", "Bias–variance", "Prediction lab", "About"]


def test_every_page_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_evaluation_controls():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Model evaluation").run()
    at.slider[0].set_value(0.2).run()
    at.radio[1].set_value("cv").run()
    assert not at.exception
    at.radio[0].set_value("BMI regression").run()
    assert not at.exception


def test_prediction_with_missing_inputs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Prediction lab").run()
    at.button[0].click().run()
    assert not at.exception
    assert any(m.label == "Predicted probability" for m in at.metric)
