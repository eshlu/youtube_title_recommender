# YouTube Title Recommender

## About
YouTube, one of the most popular platforms for user-generated content, hosts billions of videos across all genre. For creators, a video's title is arguably the single most important element determining a video’s initial performance, as it impacts click-through rate (CTR), YouTube’s recommendation algorithm, and therefore channel growth.

This project explores how video titles influence engagement by analyzing hundreds of recent uploads from top lifestyle creators. The final deliverable is a Streamlit app that recommends high-performing, thematically similar titles based on a user’s input. Under the hood, it uses a Ridge regression model trained on recent video engagement data and a TF-IDF vectorizer to capture text patterns.

## Goals
- Build a predictive model for engagement rate based on video title.
- Create an interactive Streamlit app that returns relevant, high-engagement titles.
- Deploy the trained model as an API using Google Cloud Run + Docker.
- Host the user interface on Streamlit Cloud.

## Data Collection
- Collected data using the YouTube Data API v3 from 20+ popular lifestyle/wellness creators
- Retrieved all videos publisehd in 2025 using the search.list and videos.list endpoints
- Filtered out sponsored content and incomplete records for a final list of ~400 videos

## Exploratory Data Analysis
![weekly_uploads](https://github.com/user-attachments/assets/c464dd07-c19e-4d2d-bdb8-1bfe474c31bb)

Uploads appear cyclical in nature, with noticeable peaks and valleys across weeks. This suggests that creators often post on a weekly or biweekly cadence, consistent with content calendars in the lifestyle space.

![title_lens](https://github.com/user-attachments/assets/550c5df8-0fcd-48ca-ac46-47e94f6cffb5)

Video title lengths are widely distributed, with a median of 47 characters.
This reflects both concise hooks and longer, multi-phrase titles, indicating a range of titling strategies used by top creators.

![view_v_like](https://github.com/user-attachments/assets/0c1f1e8e-14ee-4319-a523-067e1e6e3c87)

There is a clear positive correlation between views and likes, with more points lying above the diagonal, suggesting high like-to-view ratios.
Outliers—primarily sponsored content—were removed to maintain consistency.
Most videos receive under 750K views and fewer than 50K likes, highlighting a long-tail distribution.

## Preprocessing, Feature Engineering
- Cleaned video titles for standardization (lowercasing, punctuation removal)
- Computed heuristic engagement rate as:
` Engagement Rate = (Likes + Comments) / Views`
- Applied **TF-IDF vectorization** (`max_features=1000`, `stop_words='english'`)

## Modeling
- Used ridge regression to predict engagement rate from TF-IDF vectors
- Chosen for its simplicity and robustness in high-dimensional, sparse data
- Used gridsearch to tune alpha to 0.1
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
