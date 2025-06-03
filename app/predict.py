from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = FastAPI()
model = joblib.load("ridge_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

class TitleInput(BaseModel):
    title: str

def clean_title(text):
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text

@app.post("/predict")
def predict_engagement(data: TitleInput):
    try:
        clean_text = clean_title(data.title)
        vectorized = vectorizer.transform([clean_text])
        pred = model.predict(vectorized)[0]
        return {"predicted_engagement": round(pred, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))