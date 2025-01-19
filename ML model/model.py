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
                time.sleep(2)
                cdata = pytrends.interest_over_time()
                if not cdata.empty:
                    cdata['country'] = country
                    all_data = pd.concat([all_data, cdata], ignore_index=False)
            except Exception as e:
                logging.warning(f"Trends error for {country} on {single_day}: {e}")
                if "429" in str(e):
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
    """
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    all_articles = []

    COUNTRY_QUERY_VARIATIONS = {
        'KR': [
            "korea", "korea news", "korean news",
            "korea economy", "korean economy",
            "korea politics", "korea technology",
            "korea finance", "korean finance"
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
            time.sleep(1)

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
    """
    Fit a TF-IDF vectorizer + KMeans on the entire training set => save to disk.
    Return df with 'topic_id' assigned.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    cluster_vectorizer = TfidfVectorizer(max_features=3000)
    X_cluster = cluster_vectorizer.fit_transform(df['cleaned_text'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['topic_id'] = kmeans.fit_predict(X_cluster)

    # Save
    joblib.dump(cluster_vectorizer, cluster_vec_path)
    joblib.dump(kmeans, kmeans_path)

    return df


def assign_topics(df, cluster_vec_path='cluster_vectorizer.pkl', kmeans_path='kmeans_model.pkl'):
    """
    Load the previously trained cluster vectorizer + KMeans => assign 'topic_id'.
    """
    if not os.path.exists(cluster_vec_path) or not os.path.exists(kmeans_path):
        logging.error("No trained cluster model found! Please run training first.")
        return df

    cluster_vectorizer = joblib.load(cluster_vec_path)
    kmeans = joblib.load(kmeans_path)

    X = cluster_vectorizer.transform(df['cleaned_text'])
    df['topic_id'] = kmeans.predict(X)
    return df

# -------------------------------------------------------------------------
# 4) LABEL TOPIC DAY-TO-DAY TRENDS
# -------------------------------------------------------------------------
def label_topic_trends(df, date_col='date', topic_col='topic_id'):
    """
    For each (topic_id, date), count coverage => see if next day's coverage is bigger => label=1
    """
    coverage = df.groupby([topic_col, date_col]).size().reset_index(name='count')
    coverage.sort_values(by=[topic_col, date_col], inplace=True)
    coverage['next_count'] = coverage.groupby(topic_col)['count'].shift(-1)

    coverage['label'] = coverage.apply(
        lambda row: 1 if pd.notnull(row['next_count']) and row['next_count'] > row['count'] else 0,
        axis=1
    )

    out = pd.merge(df, coverage[[topic_col, date_col, 'label']], on=[topic_col, date_col], how='left')
    out['label'] = out['label'].fillna(0)
    return out

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

    start_date = datetime.now() - timedelta(days=15)
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

        # b) Label day-to-day coverage
        labeled = label_topic_trends(n_data)
        print("Label distribution:\n", labeled['label'].value_counts())

        # c) Merge Google data
        if not g_data.empty:
            g_data['date'] = pd.to_datetime(g_data['date'])
            combined = pd.merge(labeled, g_data, on=['date','country'], how='left')
        else:
            combined = labeled.copy()
            for kw in google_kw:
                combined[kw] = 0

        # fill missing numeric columns
        for kw in google_kw:
            if kw not in combined.columns:
                combined.loc[:, kw] = 0
            else:
                combined.loc[:, kw] = combined[kw].fillna(0)

        # d) Build classifier features
        clf_vectorizer = TfidfVectorizer(max_features=3000)
        X_text = clf_vectorizer.fit_transform(combined['cleaned_text'])
        joblib.dump(clf_vectorizer, "clf_vectorizer.pkl")

        google_mat = csc_matrix(combined[google_kw].values)
        X_final = hstack([X_text, google_mat]).tocsr()
        y_final = combined['label']

        # e) Class weighting
        c_counts = y_final.value_counts()
        if 1 in c_counts:
            weight_1 = float(c_counts[0]) / float(c_counts[1])
        else:
            weight_1 = 1.0
        weight_0 = 1.0
        print(f"Class Weights => 0:{weight_0}, 1:{weight_1}")
        w_tensor = torch.tensor([weight_0, weight_1])

        # f) Train the classifier
        dataset = TrendDataset(X_final, y_final)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        input_dim = X_final.shape[1]
        model = TrendPredictor(input_size=input_dim, hidden_size=128, num_classes=2)
        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_classifier(model, loader, criterion, optimizer, epochs=5)

        torch.save(model.state_dict(), "trend_predictor.pth")
        logging.info("Finished training & saved model/cluster artifacts.")

    # ------------------------- INFERENCE PIPELINE -------------------------
    else:
        # If not training, we assume the cluster model & classifier are already trained
        # 1) Assign topics with saved cluster model
        n_data = assign_topics(n_data)
        # 2) Merge google
        if not g_data.empty:
            g_data['date'] = pd.to_datetime(g_data['date'])
            combined = pd.merge(n_data, g_data, on=['date','country'], how='left')
        else:
            combined = n_data.copy()
            for kw in google_kw:
                combined[kw] = 0

        # fill missing numeric columns
        for kw in google_kw:
            if kw not in combined.columns:
                combined.loc[:, kw] = 0
            else:
                combined.loc[:, kw] = combined[kw].fillna(0)

        # 3) Build classifier features with saved clf_vectorizer
        if not os.path.exists("clf_vectorizer.pkl"):
            logging.error("No clf_vectorizer found, please train first.")
            return

        clf_vectorizer = joblib.load("clf_vectorizer.pkl")
        X_text = clf_vectorizer.transform(combined['cleaned_text'])
        google_mat = csc_matrix(combined[google_kw].values)
        X_infer = hstack([X_text, google_mat]).tocsr()

        # 4) Load classifier
        input_dim = X_infer.shape[1]
        model_infer = TrendPredictor(input_size=input_dim)
        # Use weights_only=True to avoid future unpickling warnings:
        state_dict = torch.load("trend_predictor.pth", weights_only=True)
        model_infer.load_state_dict(state_dict)
        model_infer.eval()

        # 5) Predict
        feats_tensor = torch.tensor(X_infer.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out = model_infer(feats_tensor)
            _, preds = torch.max(out, dim=1)
        combined['prediction'] = preds.numpy()

        # Show predicted=1
        trending = combined[combined['prediction'] == 1]
        print(f"Total articles: {len(combined)}")
        print(f"Predicted 1: {len(trending)}")
        print(trending[['date','title','topic_id','prediction']])

if __name__ == "__main__":
    """
    Usage:
      1) First run training (do_train=True):
         python model.py
         # respond with the country code e.g. KR
      2) Then run inference (do_train=False) to see if we get any 1 predictions:
         # Modify main(do_train=False) or run with a command line param.
    """
    # By default, let's do training in one run, or set do_train=False to skip training.
    # Here we'll show do_train=True to replicate your steps in a single script.
    main(do_train=False)

    # If you want to do inference only in a second run:
    # main(do_train=False)
