import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your preprocessed DataFrame `full_df` with required columns
# Simulating for now (replace with actual data load)
df = pd.read_csv("youtube_data_filtered.csv")

# Define heuristic engagement score
alpha = 1
df['heuristic_engagement'] = (
    df['like_count'] + (alpha * df['comment_count'])
) / df['view_count']

# Clean and vectorize titles
df['clean_title'] = df['title'].str.lower().str.replace(r'[\W_]+', ' ', regex=True)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['clean_title'])
y = df['heuristic_engagement']

# Train the model
model = Ridge(alpha=1.0)
model.fit(X, y)

# Streamlit app
st.title("YouTube Title Recommender (Engagement-Optimized)")
user_input = st.text_input("Enter a video title idea:")

if user_input:
    user_input_clean = user_input.lower()
    user_vector = vectorizer.transform([user_input_clean])
    predicted_engmt = model.predict(user_vector)[0]
    
    similarities = cosine_similarity(user_vector, X).flatten()
    df['similarity'] = similarities
    df['predicted_diff'] = np.abs(df['heuristic_engagement'] - predicted_engmt)
    
    results = df.sort_values(by=['similarity', 'predicted_diff'], ascending=[False, True])
    
    st.subheader("Top Similar High-Engagement Titles")
    st.dataframe(results[['title','view_count', 'heuristic_engagement']].head(4).sort_values('heuristic_engagement', ascending=False))