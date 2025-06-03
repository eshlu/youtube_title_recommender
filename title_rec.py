import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load data
df = pd.read_csv("youtube_data_filtered.csv")

# engagement heuristic
alpha = 1
df['heuristic_engagement'] = (
    df['like_count'] + (alpha * df['comment_count'])
) / df['view_count']

# title vectors
df['clean_title'] = df['title'].str.lower().str.replace(r'[\W_]+', ' ', regex=True)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['clean_title'])
y = df['heuristic_engagement']

# modeling
model = Ridge(alpha=1.0)
model.fit(X, y)

# streamlit 
st.set_page_config(layout="wide")
st.title("YouTube Title Recommender")
st.subheader("Enter your video concept to discover trending, high-engagement tiles from popular lifestyle creators.")
user_input = st.text_input("Enter a video title idea:")

if user_input:
    user_input_clean = user_input.lower()
    user_vector = vectorizer.transform([user_input_clean])
    predicted_engmt = model.predict(user_vector)[0]
    
    similarities = cosine_similarity(user_vector, X).flatten()
    df['similarity'] = similarities
    df['predicted_diff'] = np.abs(df['heuristic_engagement'] - predicted_engmt)
    
    results = df.sort_values(by=['similarity', 'predicted_diff'], ascending=[False, True])
    results.rename(columns={'heuristic_engagement': 'Engagement Rate', 'title':'Title', 'view_count':'View Count'}, inplace=True)
    results['Engagement Rate'] = (results['Engagement Rate'] * 100).round(1).astype(str) + '%'
    
    st.subheader("Top Similar High-Engagement Titles")
    st.dataframe(
        results[['Title', 'View Count', 'Engagement Rate']]
        .head(4)
        .sort_values('Engagement Rate', ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )
