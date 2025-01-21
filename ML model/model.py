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
# from pytrends.request import TrendReq  # <--- commented out since we're not using it
import requests  # for GDELT
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
# ENV
# -------------------------------------------------------------------------
env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------------
# 1) DATA COLLECTION
# -------------------------------------------------------------------------
# We keep the function definition in case you want it later, but we'll NOT call it.
def collect_google_trends_data(start_date, end_date, country_codes, kw_list):
    """
    Collect Google Trends data (currently not used).
    """
    """
    pytrends = TrendReq(tz=360)
    all_data = pd.DataFrame()
    date_range_list = pd.date_range(start=start_date, end=end_date, freq='D')

    for country in country_codes:
        for single_day in date_range_list:
            timeframe = f"{single_day.strftime('%Y-%m-%d')} {single_day.strftime('%Y-%m-%d')}"
            try:
                pytrends.build_payload(kw_list, timeframe=timeframe, geo=country)
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
        if 'Country' in all_data.columns:
            all_data.rename(columns={'Country': 'country'}, inplace=True)
        if 'isPartial' in all_data.columns:
            all_data.drop(columns='isPartial', inplace=True, errors='ignore')
    return all_data
    """
    return pd.DataFrame()  # Return empty so we do NOT use Google Trends
  

def _gdelt_fetch_articles(query_str, start_dt, end_dt, source_country, max_pages=4):
    """
    Helper function that fetches up to 250 * max_pages articles from GDELT
    for the given query, date/time range, and source country.
    Returns a list of articles or an empty list if error.
    """
    all_articles = []
    base_url = "http://api.gdeltproject.org/api/v2/doc/doc"

    for page in range(max_pages):  # Fetch up to max_pages
        params = {
            'query': query_str,
            'mode': 'artlist',
            'format': 'json',
            'maxrecords': '250',
            'sort': 'DateDesc',
            'startdatetime': start_dt.strftime("%Y%m%d%H%M%S"),
            'enddatetime': end_dt.strftime("%Y%m%d%H%M%S"),
            'offset': page * 250,  # Pagination offset
            'sourcecountry': source_country  # Filter by source country
        }

        headers = {
            'User-Agent': 'MyGDELTScript/1.0'
        }

        try:
            resp = requests.get(base_url, params=params, headers=headers)
            if resp.status_code != 200:
                logging.warning(f"GDELT error (status={resp.status_code}) query={query_str}: {resp.text}")
                break  # Stop fetching more pages on error
            data = resp.json()
            articles = data.get('articles', [])
            if not articles:
                break  # No more articles to fetch
            all_articles.extend(articles)
        except Exception as e:
            logging.warning(f"Exception during GDELT fetch for query={query_str}: {e}")
            break

        # Sleep to respect GDELT rate limits
        time.sleep(5)

    return all_articles


def collect_gdelt_data_by_country(start_date, end_date, country_codes):
    """
    Collect GDELT data by country code, filtering news by the source country.
    """
    all_articles = []

    # 3-day windows
    date_slices = []
    current_start = start_date
    while current_start <= end_date:
        slice_end = current_start + timedelta(days=2)  # 3-day window
        if slice_end > end_date:
            slice_end = end_date
        date_slices.append((current_start, slice_end))
        current_start = slice_end + timedelta(days=1)

    for code in country_codes:
        for (slice_start, slice_end) in date_slices:
            logging.info(f"GDELT fetch: {code} [{slice_start.strftime('%Y-%m-%d')} -> {slice_end.strftime('%Y-%m-%d')}]")
            data = _gdelt_fetch_articles(query_str="news", start_dt=slice_start, end_dt=slice_end, source_country=code)
            if not data:
                logging.warning(f"No data returned for {code} from {slice_start} to {slice_end}")
            else:
                logging.info(f"Fetched {len(data)} articles from GDELT.")
                for art in data:
                    pub_str = art.get('seendate')
                    try:
                        pub_date = datetime.strptime(pub_str, '%Y%m%dT%H%M%SZ').date() if pub_str else None
                    except ValueError:
                        logging.warning(f"Failed to parse seendate: {pub_str}")
                        pub_date = None

                    all_articles.append({
                        'country': code,
                        'date': pub_date,
                        'title': art.get('title', ''),
                        'description': art.get('extrasummary', '') or art.get('snippet', ''),
                        'source': art.get('domain', ''),
                        'url': art.get('url', '')
                    })

            # Sleep to respect GDELT's suggestion ~1 request / 5 sec
            time.sleep(5)

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
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# -------------------------------------------------------------------------
# 3) CLUSTER MODEL (TOPIC)
# -------------------------------------------------------------------------
def train_cluster_model(df, n_clusters, cluster_vec_path='cluster_vectorizer.pkl', kmeans_path='kmeans_model.pkl'):
    df = df.copy()
    cluster_vectorizer = TfidfVectorizer(max_features=3000)
    X_cluster = cluster_vectorizer.fit_transform(df['cleaned_text'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[:, 'topic_id'] = kmeans.fit_predict(X_cluster)

    joblib.dump(cluster_vectorizer, cluster_vec_path)
    joblib.dump(kmeans, kmeans_path)

    return df

def assign_topics(df, cluster_vec_path='cluster_vectorizer.pkl', kmeans_path='kmeans_model.pkl'):
    if not os.path.exists(cluster_vec_path) or not os.path.exists(kmeans_path):
        logging.error("No trained cluster model found! Please train first.")
        return df

    df = df.copy()
    cluster_vectorizer = joblib.load(cluster_vec_path)
    kmeans = joblib.load(kmeans_path)

    X = cluster_vectorizer.transform(df['cleaned_text'])
    df.loc[:, 'topic_id'] = kmeans.predict(X)
    return df

# -------------------------------------------------------------------------
# 4) LABEL TOPIC TRENDS (New Logic)
# -------------------------------------------------------------------------
def label_topic_trends(df, date_col='date', topic_col='topic_id',
                       coverage_1day=2,
                       coverage_7day=6,
                       coverage_14day=7
                       ):
    """
    Simple coverage-based logic only, ignoring Google Trends entirely.
    
    SHIFT=1 => coverage_1 >= coverage_1day
    SHIFT=7 => coverage_7 >= coverage_7day
    SHIFT=14 => coverage_14 >= coverage_14day
    """
    df = df.copy()
    df = df.sort_values(by=[topic_col, date_col])

    # Group by topic => set of unique days
    coverage_map = {}
    grouped = df.groupby(topic_col)[date_col]
    for t, dates_series in grouped:
        unique_dates = sorted(list(set(dates_series.dropna().dt.date)))
        coverage_map[t] = unique_dates

    # Prepare label columns
    df['label_shift_1'] = 0
    df['label_shift_7'] = 0
    df['label_shift_14'] = 0

    coverage_map_sets = {t: set(dlist) for t, dlist in coverage_map.items()}

    for idx, row in df.iterrows():
        t = row[topic_col]
        d = row[date_col]
        if pd.isnull(d):
            continue
        cur_date = d.date()

        # SHIFT=1 => coverage in last 4 days
        dates_1d = [(cur_date - timedelta(days=i)) for i in range(3, -1, -1)]
        coverage_1 = sum(day in coverage_map_sets[t] for day in dates_1d)

        # SHIFT=7 => coverage in last 15 days
        dates_7d = [(cur_date - timedelta(days=i)) for i in range(15, -1, -1)]
        coverage_7 = sum(day in coverage_map_sets[t] for day in dates_7d)

        # SHIFT=14 => coverage in last 30 days
        dates_14d = [(cur_date - timedelta(days=i)) for i in range(29, -1, -1)]
        coverage_14 = sum(day in coverage_map_sets[t] for day in dates_14d)

        if coverage_1 >= coverage_1day:
            df.at[idx, 'label_shift_1'] = 1
        if coverage_7 >= coverage_7day:
            df.at[idx, 'label_shift_7'] = 1
        if coverage_14 >= coverage_14day:
            df.at[idx, 'label_shift_14'] = 1

    for shift in [1, 7, 14]:
        col = f"label_shift_{shift}"
        c = df[col].value_counts().to_dict()
        logging.info(f"SHIFT={shift} label distribution: {c}")

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
        correct = 0
        total = 0
        for feats, labels in loader:
            optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy tracking
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main(do_train=True):
    """
    We'll collect 30+ days of GDELT data once, then split into:
      - Train subset: older portion
      - Inference subset: last ~7 days
    SHIFT labeling: 1-day, 7-day, 14-day windows (backward logic).
    """
    country_code = input("Enter the 2-letter country code (e.g. 'KR'): ").strip().upper()

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)  # 30 days. Adjust as needed.

    gdelt_file = "n_data.csv"

    # ----------------------------------------------------------------
    # 1) Data Gathering / Loading
    # ----------------------------------------------------------------
    if os.path.exists(gdelt_file):
        logging.info(f"Loading cached GDELT data from {gdelt_file}")
        n_data = pd.read_csv(gdelt_file)
        n_data['date'] = pd.to_datetime(n_data['date'], errors='coerce')
    else:
        # We do NOT collect Google Trends (returns empty DF)
        # g_data = collect_google_trends_data(start_date, end_date, [country_code], google_kw)
        n_data = collect_gdelt_data_by_country(start_date, end_date, [country_code])

        if n_data.empty:
            logging.error("No GDELT data returned. Exiting.")
            return

        n_data.to_csv(gdelt_file, index=False)

    n_data = preprocess_news(n_data)
    logging.info(f"After preprocess, n_data shape = {n_data.shape}")

    missing_dates = n_data['date'].isna().sum()
    logging.info(f"Number of articles with missing dates: {missing_dates}")

    cutoff_date = end_date - timedelta(days=6)  # last ~7 days
    train_data = n_data[n_data['date'] < cutoff_date].copy()
    inference_data = n_data[n_data['date'] >= cutoff_date].copy()

    logging.info(f"Train data shape = {train_data.shape}, Inference data shape = {inference_data.shape}")

    # ----------------------------------------------------------------
    # 2) Training
    # ----------------------------------------------------------------
    if do_train:
        if train_data.empty:
            logging.error("No data in the training subset. Exiting.")
            return

        # 2.1) Cluster the training data
        train_data = train_cluster_model(train_data, n_clusters=20)

        # 2.2) Label training data with coverage-only logic
        labeled_train = label_topic_trends(train_data)

        # 2.3) Vectorize ONLY the GDELT text for SHIFT=1
        vec_1day = TfidfVectorizer(max_features=3000)
        X_text_1 = vec_1day.fit_transform(labeled_train['cleaned_text'])
        joblib.dump(vec_1day, "clf_vectorizer_1day.pkl")

        X_1_final = X_text_1  # just text
        y_1_final = labeled_train['label_shift_1']
        c_1 = y_1_final.value_counts().to_dict()

        c0_1 = c_1.get(0, 0)
        c1_1 = c_1.get(1, 0)
        if c0_1 == 0 or c1_1 == 0:
            w_1 = 1.0
        else:
            w_1 = float(c0_1) / float(c1_1)
        w_1_tensor = torch.tensor([1.0, w_1], dtype=torch.float32)

        ds_1 = TrendDataset(X_1_final, y_1_final)
        loader_1 = DataLoader(ds_1, batch_size=32, shuffle=True)

        model_1 = TrendPredictor(input_size=X_1_final.shape[1], hidden_size=128, num_classes=2)
        crit_1 = nn.CrossEntropyLoss(weight=w_1_tensor)
        opt_1 = torch.optim.Adam(model_1.parameters(), lr=0.001)

        print(f"\nTraining SHIFT=1 classifier. SHIFT=1 label distribution: {c_1}")
        train_classifier(model_1, loader_1, crit_1, opt_1, epochs=10)
        torch.save(model_1.state_dict(), "trend_predictor_1day.pth")

        # SHIFT=7
        vec_7 = TfidfVectorizer(max_features=3000)
        X_text_7 = vec_7.fit_transform(labeled_train['cleaned_text'])
        joblib.dump(vec_7, "clf_vectorizer_7days.pkl")

        X_7_final = X_text_7
        y_7_final = labeled_train['label_shift_7']
        c_7 = y_7_final.value_counts().to_dict()

        c0_7 = c_7.get(0, 0)
        c1_7 = c_7.get(1, 0)
        if c0_7 == 0 or c1_7 == 0:
            w_7 = 1.0
        else:
            w_7 = float(c0_7) / float(c1_7)
        w_7_tensor = torch.tensor([1.0, w_7], dtype=torch.float32)

        ds_7 = TrendDataset(X_7_final, y_7_final)
        loader_7 = DataLoader(ds_7, batch_size=32, shuffle=True)

        model_7 = TrendPredictor(input_size=X_7_final.shape[1], hidden_size=128, num_classes=2)
        crit_7 = nn.CrossEntropyLoss(weight=w_7_tensor)
        opt_7 = torch.optim.Adam(model_7.parameters(), lr=0.001)

        print(f"\nTraining SHIFT=7 classifier. SHIFT=7 label distribution: {c_7}")
        train_classifier(model_7, loader_7, crit_7, opt_7, epochs=10)
        torch.save(model_7.state_dict(), "trend_predictor_7days.pth")

        # SHIFT=14
        vec_14 = TfidfVectorizer(max_features=3000)
        X_text_14 = vec_14.fit_transform(labeled_train['cleaned_text'])
        joblib.dump(vec_14, "clf_vectorizer_14days.pkl")

        X_14_final = X_text_14
        y_14_final = labeled_train['label_shift_14']
        c_14 = y_14_final.value_counts().to_dict()

        c0_14 = c_14.get(0, 0)
        c1_14 = c_14.get(1, 0)
        if c0_14 == 0 or c1_14 == 0:
            w_14 = 1.0
        else:
            w_14 = float(c0_14) / float(c1_14)
        w_14_tensor = torch.tensor([1.0, w_14], dtype=torch.float32)

        ds_14 = TrendDataset(X_14_final, y_14_final)
        loader_14 = DataLoader(ds_14, batch_size=32, shuffle=True)

        model_14 = TrendPredictor(input_size=X_14_final.shape[1], hidden_size=128, num_classes=2)
        crit_14 = nn.CrossEntropyLoss(weight=w_14_tensor)
        opt_14 = torch.optim.Adam(model_14.parameters(), lr=0.001)

        print(f"\nTraining SHIFT=14 classifier. SHIFT=14 label distribution: {c_14}")
        train_classifier(model_14, loader_14, crit_14, opt_14, epochs=10)
        torch.save(model_14.state_dict(), "trend_predictor_14days.pth")

        logging.info("Finished training. Models for shift=1,7,14 saved.")

    else:
        # ----------------------------------------------------------------
        # INFERENCE on the last ~7 days of data
        # ----------------------------------------------------------------
        if inference_data.empty:
            logging.warning("No data in the last 7 days. Nothing to predict.")
            return

        # Assign topics to the inference set using the trained cluster model
        inference_data = assign_topics(inference_data)

        combined_infer = inference_data.copy()

        # SHIFT=1 Inference
        if not os.path.exists("clf_vectorizer_1day.pkl") or not os.path.exists("trend_predictor_1day.pth"):
            logging.error("1-day model files not found. Please train first.")
            return
        vec_1day = joblib.load("clf_vectorizer_1day.pkl")
        X_1_infer_text = vec_1day.transform(combined_infer['cleaned_text'])

        model_1_infer = TrendPredictor(input_size=X_1_infer_text.shape[1], hidden_size=128, num_classes=2)
        st_1 = torch.load("trend_predictor_1day.pth")
        model_1_infer.load_state_dict(st_1)
        model_1_infer.eval()

        feats_1 = torch.tensor(X_1_infer_text.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out_1 = model_1_infer(feats_1)
            probs_1 = torch.softmax(out_1, dim=1)[:,1].numpy()
            _, pred_1 = torch.max(out_1, dim=1)

        combined_infer['prediction_1day'] = pred_1.numpy()
        combined_infer['prob_1day'] = probs_1

        # SHIFT=7 Inference
        if not os.path.exists("clf_vectorizer_7days.pkl") or not os.path.exists("trend_predictor_7days.pth"):
            logging.error("7-day model files not found. Please train first.")
            return
        vec_7 = joblib.load("clf_vectorizer_7days.pkl")
        X_7_infer_text = vec_7.transform(combined_infer['cleaned_text'])

        model_7_infer = TrendPredictor(input_size=X_7_infer_text.shape[1], hidden_size=128, num_classes=2)
        st_7 = torch.load("trend_predictor_7days.pth")
        model_7_infer.load_state_dict(st_7)
        model_7_infer.eval()

        feats_7 = torch.tensor(X_7_infer_text.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out_7 = model_7_infer(feats_7)
            probs_7 = torch.softmax(out_7, dim=1)[:,1].numpy()
            _, pred_7 = torch.max(out_7, dim=1)

        combined_infer['prediction_7days'] = pred_7.numpy()
        combined_infer['prob_7days'] = probs_7

        # SHIFT=14 Inference
        if not os.path.exists("clf_vectorizer_14days.pkl") or not os.path.exists("trend_predictor_14days.pth"):
            logging.error("14-day model files not found. Please train first.")
            return
        vec_14 = joblib.load("clf_vectorizer_14days.pkl")
        X_14_infer_text = vec_14.transform(combined_infer['cleaned_text'])

        model_14_infer = TrendPredictor(input_size=X_14_infer_text.shape[1], hidden_size=128, num_classes=2)
        st_14 = torch.load("trend_predictor_14days.pth")
        model_14_infer.load_state_dict(st_14)
        model_14_infer.eval()

        feats_14 = torch.tensor(X_14_infer_text.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out_14 = model_14_infer(feats_14)
            probs_14 = torch.softmax(out_14, dim=1)[:,1].numpy()
            _, pred_14 = torch.max(out_14, dim=1)

        combined_infer['prediction_14days'] = pred_14.numpy()
        combined_infer['prob_14days'] = probs_14

        # Show overall counts
        trending_1 = combined_infer[combined_infer['prediction_1day'] == 1]
        trending_7 = combined_infer[combined_infer['prediction_7days'] == 1]
        trending_14 = combined_infer[combined_infer['prediction_14days'] == 1]

        print(f"\nTotal articles in last 7 days: {len(combined_infer)}")
        print(f"Predicted trending (tomorrow, SHIFT=1): {len(trending_1)}")
        print(f"Predicted trending (next week, SHIFT=7): {len(trending_7)}")
        print(f"Predicted trending (next 2 weeks, SHIFT=14): {len(trending_14)}\n")

        # Top 5 articles for each window by probability
        trending_1_sorted = trending_1.sort_values(by='prob_1day', ascending=False)
        top_5_1 = trending_1_sorted.head(5)
        print(f"Out of {len(trending_1)} estimated trending articles for SHIFT=1, here are the top 5:")
        for i, row in top_5_1.iterrows():
            print(f"- {row['title']} (prob={row['prob_1day']:.4f})")

        trending_7_sorted = trending_7.sort_values(by='prob_7days', ascending=False)
        top_5_7 = trending_7_sorted.head(5)
        print(f"\nOut of {len(trending_7)} estimated trending articles for SHIFT=7, here are the top 5:")
        for i, row in top_5_7.iterrows():
            print(f"- {row['title']} (prob={row['prob_7days']:.4f})")

        trending_14_sorted = trending_14.sort_values(by='prob_14days', ascending=False)
        top_5_14 = trending_14_sorted.head(5)
        print(f"\nOut of {len(trending_14)} estimated trending articles for SHIFT=14, here are the top 5:")
        for i, row in top_5_14.iterrows():
            print(f"- {row['title']} (prob={row['prob_14days']:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Trend Predictor with SHIFT=1,7,14 (GDELT only)')
    parser.add_argument('--train', action='store_true', help='Train the models')
    args = parser.parse_args()

    if args.train:
        main(do_train=True)
    else:
        main(do_train=False)
