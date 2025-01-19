import os
import re
import time
import logging
import joblib
from datetime import datetime, timedelta

import nltk
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytrends.request import TrendReq
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import hstack, csc_matrix

from dotenv import load_dotenv
from pathlib import Path

try:
    from langdetect import detect
except ImportError:
    raise ImportError("Please install langdetect via `pip install langdetect`")

logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------
# ENV & API KEYS
# -------------------------------------------------------------------------
env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    logging.error("NewsAPI key missing. Please set it in .env or environment.")
    exit()

# -------------------------------------------------------------------------
# 1) DATA COLLECTION
# -------------------------------------------------------------------------
def collect_google_trends_data(start_date, end_date, country_codes, kw_list):
    pytrends = TrendReq(tz=360)
    all_data = pd.DataFrame()
    date_range_list = pd.date_range(start=start_date, end=end_date, freq='D')

    for country in country_codes:
        for single_day in date_range_list:
            timeframe = f"{single_day.strftime('%Y-%m-%d')} {single_day.strftime('%Y-%m-%d')}"
            try:
                pytrends.build_payload(kw_list, timeframe=timeframe, geo=country)
                time.sleep(2)  # small delay
                cdata = pytrends.interest_over_time()
                if not cdata.empty:
                    cdata['country'] = country
                    all_data = pd.concat([all_data, cdata], ignore_index=False)
            except Exception as e:
                logging.warning(f"Trends error for {country} on {single_day}: {e}")
                if "429" in str(e):
                    logging.warning("Rate limit encountered. Sleeping 60s...")
                    time.sleep(60)
                continue

    if not all_data.empty:
        all_data.reset_index(inplace=True)
        # rename 'Country' -> 'country' if needed
        if 'Country' in all_data.columns:
            all_data.rename(columns={'Country': 'country'}, inplace=True)
        if 'isPartial' in all_data.columns:
            all_data.drop(columns='isPartial', inplace=True, errors='ignore')
    return all_data


def collect_newsapi_data_by_country(start_date, end_date, country_codes):
    """
    Multiple queries for each country => each query gets up to 100 articles
    on page=1 only (avoid free-tier limit error). 
    Max 30 days allowed for from_param / to with a free-tier account.
    """
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    all_articles = []

    COUNTRY_QUERY_VARIATIONS = {
        'KR': [
            "korea", "korea news", "korean news",
            "korea economy", "korean economy",
            "korea politics", "korean politics",
            "korea technology", "korean technology",
            "korea finance", "korean finance",
            "korean"
        ],
        'JP': ["japan", "japan news", "japan economy", "japan politics"],
        'US': ["united states", "us news", "us economy", "us politics"],
        'CN': ["china", "china news", "china economy", "china politics"],
        'GB': ["united kingdom", "uk news", "uk economy", "uk politics"],
    }

    for code in country_codes:
        queries = COUNTRY_QUERY_VARIATIONS.get(code.upper(), [code])
        for q in queries:
            try:
                resp = newsapi.get_everything(
                    q=q,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy',
                    page_size=100,
                    page=1
                )
            except Exception as e:
                logging.warning(f"NewsAPI error for query={q}: {e}")
                continue

            if resp.get('status') != 'ok':
                logging.warning(f"NewsAPI status {resp.get('status')} for query={q}")
                continue

            articles = resp.get('articles', [])
            for art in articles:
                pub_str = art.get('publishedAt')
                try:
                    pub_date = datetime.strptime(pub_str, '%Y-%m-%dT%H:%M:%SZ').date() if pub_str else None
                except ValueError:
                    pub_date = None

                all_articles.append({
                    'country': code,
                    'date': pub_date,
                    'title': art.get('title', ''),
                    'description': art.get('description', ''),
                    'source': art['source']['name'] if art.get('source') else '',
                    'url': art.get('url', '')
                })
            time.sleep(1)  # small delay

    df = pd.DataFrame(all_articles)
    df.drop_duplicates(subset=['url'], inplace=True)
    return df

# -------------------------------------------------------------------------
# 2) PREPROCESS
# -------------------------------------------------------------------------
def clean_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = re.sub(r'http\S+|@\w+|#[^\s]+', '', txt)
    txt = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', '', txt)
    txt = txt.strip().lower()

    try:
        lng = detect(txt)
    except:
        lng = 'en'

    nltk.download('stopwords', quiet=True)
    try:
        sw = set(nltk.corpus.stopwords.words(lng))
    except OSError:
        sw = set(nltk.corpus.stopwords.words('english'))

    tokens = [w for w in txt.split() if w not in sw]
    return ' '.join(tokens)


def preprocess_news(df):
    df = df.copy()
    if 'title' in df.columns and 'description' in df.columns:
        df['content'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    else:
        df['content'] = df.get('content', '').fillna('')

    df['cleaned_text'] = df['content'].apply(clean_text)
    df['date'] = pd.to_datetime(df['date'])
    return df

# -------------------------------------------------------------------------
# 3) CLUSTER MODEL (TOPIC)
# -------------------------------------------------------------------------
def train_cluster_model(df, n_clusters=10, cluster_vec_path='cluster_vectorizer.pkl', kmeans_path='kmeans_model.pkl'):
    cluster_vectorizer = TfidfVectorizer(max_features=3000)
    X_cluster = cluster_vectorizer.fit_transform(df['cleaned_text'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['topic_id'] = kmeans.fit_predict(X_cluster)

    # Save
    joblib.dump(cluster_vectorizer, cluster_vec_path)
    joblib.dump(kmeans, kmeans_path)

    return df


def assign_topics(df, cluster_vec_path='cluster_vectorizer.pkl', kmeans_path='kmeans_model.pkl'):
    if not os.path.exists(cluster_vec_path) or not os.path.exists(kmeans_path):
        logging.error("No trained cluster model found! Please run training first.")
        return df

    cluster_vectorizer = joblib.load(cluster_vec_path)
    kmeans = joblib.load(kmeans_path)

    X = cluster_vectorizer.transform(df['cleaned_text'])
    df['topic_id'] = kmeans.predict(X)
    return df

# -------------------------------------------------------------------------
# 4) LABEL TOPIC TRENDS (7 / 14 DAYS)
# -------------------------------------------------------------------------
def label_topic_trends(df, date_col='date', topic_col='topic_id', shifts=[7,14]):
    """
    For each (topic_id, date), count coverage => see if coverage increases after 'shift' days.
    Create separate labels for shift=7 and shift=14.
    """
    for shift in shifts:
        coverage = df.groupby([topic_col, date_col]).size().reset_index(name='count')
        coverage.sort_values(by=[topic_col, date_col], inplace=True)
        coverage[f'next_count_shift_{shift}'] = coverage.groupby(topic_col)['count'].shift(-shift)

        coverage[f'label_shift_{shift}'] = coverage.apply(
            lambda row: 1 if pd.notnull(row[f'next_count_shift_{shift}']) and row[f'next_count_shift_{shift}'] > row['count'] else 0,
            axis=1
        )

        df = pd.merge(df, coverage[[topic_col, date_col, f'label_shift_{shift}']], on=[topic_col, date_col], how='left')

    for shift in shifts:
        df[f'label_shift_{shift}'] = df[f'label_shift_{shift}'].fillna(0).astype(int)

    return df

# -------------------------------------------------------------------------
# 5) DATASET
# -------------------------------------------------------------------------
class TrendDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row_x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        return row_x, torch.tensor(self.y[idx], dtype=torch.long)

# -------------------------------------------------------------------------
# 6) CLASSIFIER
# -------------------------------------------------------------------------
class TrendPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=2):
        super(TrendPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------------------------------------------------------
# 7) TRAINING LOOP
# -------------------------------------------------------------------------
def train_classifier(model, loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for feats, labels in loader:
            optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg:.4f}")

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main(do_train=True):
    country_code = input("Enter the 2-letter country code (e.g. 'KR'): ")

    # Collect up to 30 days of data
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now() - timedelta(days=1)

    # 1) Collect Data
    google_kw = ['technology', 'economy', 'health', 'politics']
    g_data = collect_google_trends_data(start_date, end_date, [country_code], google_kw)
    n_data = collect_newsapi_data_by_country(start_date, end_date, [country_code])
    if n_data.empty:
        logging.error("No NewsAPI data. Exiting.")
        return

    # 2) Preprocess
    n_data = preprocess_news(n_data)

    # ------------------------- TRAINING PIPELINE -------------------------
    if do_train:
        # a) Train cluster model => assign topics
        n_data = train_cluster_model(n_data, n_clusters=10)

        # b) Label coverage for 7 and 14 days
        labeled = label_topic_trends(n_data, shifts=[7,14])
        print("Label distribution for 7-day shift:")
        print(labeled['label_shift_7'].value_counts())
        print("\nLabel distribution for 14-day shift:")
        print(labeled['label_shift_14'].value_counts())

        # c) Merge Google data
        if not g_data.empty:
            g_data['date'] = pd.to_datetime(g_data['date'])
            combined = pd.merge(labeled, g_data, on=['date','country'], how='left')
        else:
            combined = labeled.copy()
            for kw in google_kw:
                combined.loc[:, kw] = 0

        for kw in google_kw:
            if kw not in combined.columns:
                combined.loc[:, kw] = 0
            else:
                combined.loc[:, kw] = combined[kw].fillna(0)

        # d) Build classifier features for SHIFT=7
        vec_7 = TfidfVectorizer(max_features=3000)
        X_text_7 = vec_7.fit_transform(combined['cleaned_text'])
        joblib.dump(vec_7, "clf_vectorizer_7days.pkl")

        google_mat_7 = csc_matrix(combined[google_kw].values)
        X_final_7 = hstack([X_text_7, google_mat_7]).tocsr()
        y_final_7 = combined['label_shift_7']

        # e) Class weighting for SHIFT=7
        c_counts_7 = y_final_7.value_counts()
        if 1 in c_counts_7:
            weight_1_7 = float(c_counts_7[0]) / float(c_counts_7[1])
        else:
            weight_1_7 = 1.0
        weight_0_7 = 1.0
        print(f"Class Weights (7 days) => 0:{weight_0_7}, 1:{weight_1_7}")
        w_tensor_7 = torch.tensor([weight_0_7, weight_1_7])

        # f) Train classifier for SHIFT=7
        ds_7 = TrendDataset(X_final_7, y_final_7)
        loader_7 = DataLoader(ds_7, batch_size=32, shuffle=True)

        input_dim_7 = X_final_7.shape[1]
        model_7 = TrendPredictor(input_size=input_dim_7, hidden_size=128, num_classes=2)
        criterion_7 = nn.CrossEntropyLoss(weight=w_tensor_7)
        optimizer_7 = torch.optim.Adam(model_7.parameters(), lr=0.001)
        print("\nTraining Classifier (7 days):")
        train_classifier(model_7, loader_7, criterion_7, optimizer_7, epochs=5)
        torch.save(model_7.state_dict(), "trend_predictor_7days.pth")

        # g) Build classifier features for SHIFT=14
        vec_14 = TfidfVectorizer(max_features=3000)
        X_text_14 = vec_14.fit_transform(combined['cleaned_text'])
        joblib.dump(vec_14, "clf_vectorizer_14days.pkl")

        google_mat_14 = csc_matrix(combined[google_kw].values)
        X_final_14 = hstack([X_text_14, google_mat_14]).tocsr()
        y_final_14 = combined['label_shift_14']

        c_counts_14 = y_final_14.value_counts()
        if 1 in c_counts_14:
            weight_1_14 = float(c_counts_14[0]) / float(c_counts_14[1])
        else:
            weight_1_14 = 1.0
        weight_0_14 = 1.0
        print(f"\nClass Weights (14 days) => 0:{weight_0_14}, 1:{weight_1_14}")
        w_tensor_14 = torch.tensor([weight_0_14, weight_1_14])

        ds_14 = TrendDataset(X_final_14, y_final_14)
        loader_14 = DataLoader(ds_14, batch_size=32, shuffle=True)

        input_dim_14 = X_final_14.shape[1]
        model_14 = TrendPredictor(input_size=input_dim_14, hidden_size=128, num_classes=2)
        criterion_14 = nn.CrossEntropyLoss(weight=w_tensor_14)
        optimizer_14 = torch.optim.Adam(model_14.parameters(), lr=0.001)
        print("\nTraining Classifier (14 days):")
        train_classifier(model_14, loader_14, criterion_14, optimizer_14, epochs=5)
        torch.save(model_14.state_dict(), "trend_predictor_14days.pth")

        logging.info("Finished training & saved all models/cluster artifacts.")

    # ------------------------- INFERENCE PIPELINE -------------------------
    else:
        # Load cluster model and assign topics
        n_data = assign_topics(n_data)

        # Merge Google data
        if not g_data.empty:
            g_data['date'] = pd.to_datetime(g_data['date'])
            combined = pd.merge(n_data, g_data, on=['date','country'], how='left')
        else:
            combined = n_data.copy()
            for kw in google_kw:
                combined.loc[:, kw] = 0

        for kw in google_kw:
            if kw not in combined.columns:
                combined.loc[:, kw] = 0
            else:
                combined.loc[:, kw] = combined[kw].fillna(0)

        # 1) Load classifiers & vectorizers for 7-day and 14-day
        if not os.path.exists("clf_vectorizer_7days.pkl") or not os.path.exists("trend_predictor_7days.pth"):
            logging.error("7-day model files not found. Please train first.")
            return
        if not os.path.exists("clf_vectorizer_14days.pkl") or not os.path.exists("trend_predictor_14days.pth"):
            logging.error("14-day model files not found. Please train first.")
            return

        vec_7 = joblib.load("clf_vectorizer_7days.pkl")
        vec_14 = joblib.load("clf_vectorizer_14days.pkl")

        # 2) Features for SHIFT=7
        X_text_7_infer = vec_7.transform(combined['cleaned_text'])
        google_mat_7_infer = csc_matrix(combined[google_kw].values)
        X_infer_7 = hstack([X_text_7_infer, google_mat_7_infer]).tocsr()

        model_7_infer = TrendPredictor(input_size=X_infer_7.shape[1], hidden_size=128, num_classes=2)
        st_7 = torch.load("trend_predictor_7days.pth", weights_only=True)
        model_7_infer.load_state_dict(st_7)
        model_7_infer.eval()

        feats_7_tensor = torch.tensor(X_infer_7.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out_7 = model_7_infer(feats_7_tensor)
            _, pred_7 = torch.max(out_7, dim=1)

        combined['prediction_7days'] = pred_7.numpy()

        # 3) Features for SHIFT=14
        X_text_14_infer = vec_14.transform(combined['cleaned_text'])
        google_mat_14_infer = csc_matrix(combined[google_kw].values)
        X_infer_14 = hstack([X_text_14_infer, google_mat_14_infer]).tocsr()

        model_14_infer = TrendPredictor(input_size=X_infer_14.shape[1], hidden_size=128, num_classes=2)
        st_14 = torch.load("trend_predictor_14days.pth", weights_only=True)
        model_14_infer.load_state_dict(st_14)
        model_14_infer.eval()

        feats_14_tensor = torch.tensor(X_infer_14.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out_14 = model_14_infer(feats_14_tensor)
            _, pred_14 = torch.max(out_14, dim=1)

        combined['prediction_14days'] = pred_14.numpy()

        # 4) Show predictions
        trending_7 = combined[combined['prediction_7days'] == 1]
        trending_14 = combined[combined['prediction_14days'] == 1]

        print(f"\nTotal articles: {len(combined)}")
        print(f"Predicted 1 for 7 Days: {len(trending_7)}")
        print(f"Predicted 1 for 14 Days: {len(trending_14)}\n")

        print("Trending Articles for 1 Week Later:")
        print(trending_7[['date','title','topic_id','prediction_7days']])

        print("\nTrending Articles for 2 Weeks Later:")
        print(trending_14[['date','title','topic_id','prediction_14days']])

# -------------------------------------------------------------------------
# RUN SCRIPT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage:
      1) Train:
         python model.py --train
      2) Inference:
         python model.py
    """
    import argparse

    parser = argparse.ArgumentParser(description='Trend Predictor')
    parser.add_argument('--train', action='store_true', help='Set to train the models')
    args = parser.parse_args()

    if args.train:
        main(do_train=True)
    else:
        main(do_train=False)
