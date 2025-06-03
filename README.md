# YouTube Title Recommender

## About
This project helps lifestyle & wellness content creators discover video titles that are both on-trend and optimized for engagement. After collecting titles from popular YouTubers, a model is trained to predict engagement based on title text, and the deployed recommender app allows users to input a video idea and receive high-performing title suggestions.

## Goals
- Build a predictive model for engagement rate based on video title.
- Create an interactive Streamlit app that returns relevant, high-engagement titles.
- Deploy the trained model as an API using Google Cloud Run + Docker.
- Host the user interface on Streamlit Cloud.

## Data Collection
- Collected data using the YouTube Data API v3 from 20+ popular lifestyle/wellness creators
- Retrieved all videos publisehd in 2025 using the search.list and videos.list endpoints
- Filtered out sponsored content and incomplete records for a final list of ~400 videos

## Preprocessing, Feature Engineering
- Cleaned video titles for standardization (lowercasing, punctuation removal)
- Computed heuristic engagement rate as:
` Engagement Rate = (Likes + Comments) / Views`
- Vectorized titles using TF-IDF

## Modeling
- Used ridge regression to predict engagement rate from TF-IDF vectors
- Achieved RMSE = 0.026 (good predictive performance)
- Used cosine similarity to compare user-input title to training data

## App workflow
- User enters a video concept (e.g. "morning routine", "weekend vlog")
- App sends it to the cloud-hosted model API
- Model predicts expected engagement
- App compares title to dataset and recommends the 4 most similar, higher engagement titles

## Model API
- predict.py contains a Flask API with a single POST `/predict` endpoint
- API is deployed on Google Cloud Run, containerized using Docker
- Model + vectorizer are stored as .joblib files

## Examples
### API
`curl -X POST https://youtube-model-api-836750386700.us-west1.run.app/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "skincare routine"}'`

Expected response (random engagement for demo purposes):

`{"predicted_engagement": 0.0451}`

### Streamlit interface
[Link to Streamlit App](shl-418-youtube.streamlit.app)

## Deployment 
| Component        | Platform         |   
|------------------|------------------|
| Model API        | Google Cloud Run |
| Frontend App     | Streamlit Cloud  |
| Containerization | Docker           |

## Directory Structure
```
├── app/
│   ├── title_rec.py          
│   ├── predict.py            
│   ├── ridge_model.joblib    
│   ├── tfidf_vectorizer.joblib
│   └── youtube_data_filtered.csv
├── Dockerfile
├── requirements.txt
├── README.md
└── presentation.pdf
```


Data collected via the YouTube Data API for academic and non-commercial purposes only. 
