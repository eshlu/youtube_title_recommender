import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import joblib

df = pd.read_csv("youtube_data_filtered.csv")
df['heuristic_engagement'] = (df['like_count'] + df['comment_count']) / df['view_count']
df['clean_title'] = df['title'].str.lower().str.replace(r'[\W_]+', ' ', regex=True)

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['clean_title'])
y = df['heuristic_engagement']

model = Ridge(alpha=0.1)
model.fit(X, y)

joblib.dump(model, 'ridge_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')