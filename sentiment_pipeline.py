"""Five-level review rating baseline; no NLTK downloads or hosted models."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def evaluate(frame):
    if not {"Text","Score"} <= set(frame):raise ValueError("Expected Text and Score columns")
    frame=frame.dropna(subset=["Text","Score"]).drop_duplicates(subset=["Text","Score"])
    if frame.groupby("Text")["Score"].nunique().gt(1).any():raise ValueError("Conflicting duplicate review labels")
    train,test=train_test_split(frame,test_size=.25,stratify=frame["Score"],random_state=42)
    model=Pipeline([("tfidf",TfidfVectorizer(stop_words="english",max_features=20000)),("classifier",LogisticRegression(max_iter=1000))])
    model.fit(train["Text"],train["Score"])
    return model, classification_report(test["Score"],model.predict(test["Text"]),zero_division=0), train, test
