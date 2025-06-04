# YouTube Title Recommender — App Directory

This folder contains the full codebase for the deployed Streamlit app, which recommends high-engagement YouTube titles based on user-provided concepts.

## What’s Inside
- `starter_script.ipynb` — Jupyter notebook to collect video data from selected channels via the YouTube Data API (**Note:** API key not included).
- - `youtube_data_filtered.csv` — Cleaned dataset of YouTube video metadata from 20+ lifestyle/wellness creators
- `train.py` — script for:
  - Loading cleaned YouTube video data
  - Preprocessing and vectorizing video titles
  - Training a Ridge regression model on a heuristic engagement score
  - Saving both the model and vectorizer for later use in the prediction API
- `ridge_model.joblib` — Trained Ridge regression model predicting engagement rate from video title text.
- `tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer used for text preprocessing.
- `predict.py` — FastAPI app that loads the trained Ridge regression model and TF-IDF vectorizer, and serves predictions via a `/predict` POST endpoint.
- `title_rec.py` — Main Streamlit app script. Loads the trained model, takes user input, calls the model API, and displays top recommended titles with predicted engagement rates.


## App Overview

The Streamlit app recommends video titles by:
1. Taking a user-provided concept or idea.
2. Sending the title to a **deployed model API** (hosted on Google Cloud Run via Docker).
3. Predicting expected engagement rate.
4. Comparing the concept to real video titles using **cosine similarity**.
5. Returning the most relevant and high-performing titles from the dataset.

## Deployment 

The model API is Dockerized and designed to be deployed to Google Cloud Run. Be sure to:
	•	Expose port 8080 in your Dockerfile.
	•	Push your container image to Google Artifact Registry.
	•	Deploy via gcloud run deploy.
 
The app is deployed on [Streamlit Cloud]([https://streamlit.io/](https://shl-418-youtube.streamlit.app/)) and connects to a backend model API deployed on Google Cloud Run. 

## ⚙️ How to Run Locally

Clone the repo and from the project root:

```bash
cd app
pip install -r requirements.txt
streamlit run title_rec.py
```
