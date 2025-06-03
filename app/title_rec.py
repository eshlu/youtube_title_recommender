import streamlit as st
import pandas as pd
import requests

df = pd.read_csv("app/youtube_data_filtered.csv")
df['heuristic_engagement'] = (df['like_count'] + df['comment_count']) / df['view_count']
df['clean_title'] = df['title'].str.lower().str.replace(r'[\W_]+', ' ', regex=True)

st.set_page_config(layout="wide")
st.title("YouTube Title Recommender")
st.subheader("Enter your video concept to discover trending, high-engagement tiles from popular lifestyle creators.")
user_input = st.text_input("Your video concept")

def get_engagement_from_api(title):
    response = requests.post("https://youtube-model-api-836750386700.us-west1.run.app/predict", json={"title": title})
    if response.status_code == 200:
        return response.json()["predicted_engagement"]
    else:
        return None

if user_input:
    predicted_engmt = get_engagement_from_api(user_input.lower())
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import joblib
    import numpy as np
    
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    X = vectorizer.transform(df['clean_title'])
    user_vector = vectorizer.transform([user_input.lower()])
    
    similarities = cosine_similarity(user_vector, X).flatten()
    df['similarity'] = similarities
    df['predicted_diff'] = np.abs(df['heuristic_engagement'] - predicted_engmt)
    
    results = df.sort_values(by=['similarity', 'predicted_diff'], ascending=[False, True])
    results = results[['title', 'view_count', 'heuristic_engagement']].head(4).sort_values('heuristic_engagement', ascending=False)
    results.rename(columns={'heuristic_engagement': 'Engagement Rate', 'title':'Title', 'view_count':'View Count'}, inplace=True)
    results['Engagement Rate'] = (results['Engagement Rate'] * 100).round(1).astype(str) + '%'
    
    st.subheader("Top Similar High-Engagement Titles")
    st.dataframe(results.reset_index(drop=True), use_container_width=True)
