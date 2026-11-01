import pandas as pd
from sentiment_pipeline import evaluate
def test_pipeline_splits_before_fit_and_keeps_reviews_disjoint():
    frame=pd.read_csv("data/synthetic_reviews.csv")
    _,report,train,test=evaluate(frame)
    assert set(train["Text"]).isdisjoint(set(test["Text"]))
    assert "macro avg" in report
