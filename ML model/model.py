import os
import re
import time
import logging
import joblib
from datetime import datetime, timedelta
from pathlib import Path

import nltk
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from dotenv import load_dotenv
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
# FETCH ARTICLES
# -------------------------------------------------------------------------
def gdelt_fetch_articles_noquery_4weeks(country_code):
    """
    Fetch articles from GDELT for each of the last 30 days, splitting
    each day into 3 eight-hour windows. Each request fetches up to 250 articles.
    """
    base_url = "http://api.gdeltproject.org/api/v2/doc/doc"
    all_articles = []
    now_utc = datetime.utcnow()

    for i in range(30):
        day_end = now_utc - timedelta(days=i)
        day_start = day_end - timedelta(days=1)

        chunk_windows = [
            (day_start, day_start + timedelta(hours=8)),
            (day_start + timedelta(hours=8), day_start + timedelta(hours=16)),
            (day_start + timedelta(hours=16), day_end)
        ]

        for chunk_idx, (st, en) in enumerate(chunk_windows, start=1):
            start_str = st.strftime("%Y%m%d%H%M%S")
            end_str = en.strftime("%Y%m%d%H%M%S")

            params = {
                'query': f'sourcecountry:{country_code}',
                'mode': 'artlist',
                'format': 'json',
                'maxrecords': '250',
                'sort': 'DateDesc',
                'startdatetime': start_str,
                'enddatetime': end_str,
            }
            headers = {'User-Agent': 'MyGDELTScript/1.0'}

            try:
                resp = requests.get(base_url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    articles = data.get('articles', [])
                    logging.info(f"[{country_code}] Day {i+1}, chunk {chunk_idx}/3:"
                                 f" {start_str}->{end_str}, got {len(articles)} articles.")
                    if not articles:
                        continue
                    all_articles.extend(articles)
                else:
                    logging.warning(f"GDELT error (status={resp.status_code}): {resp.text}")
                    break
            except Exception as e:
                logging.warning(f"Exception during GDELT fetch: {e}")
                break

            time.sleep(5)

    return all_articles

# -------------------------------------------------------------------------
# PREPROCESS
# -------------------------------------------------------------------------
def clean_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = re.sub(r'http\S+|@\w+|#[^\s]+', '', txt)  # remove URLs, mentions, hashtags
    txt = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', '', txt)  # remove non-alpha except accented
    txt = txt.strip().lower()

    # Language detection
    try:
        lng = detect(txt)
    except:
        lng = 'en'  # default to English if detection fails

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
# CLUSTER MODEL (TOPIC)
# -------------------------------------------------------------------------
def train_cluster_model(df, n_clusters,
                        cluster_vec_path,
                        kmeans_path):
    df = df.copy()
    cluster_vectorizer = TfidfVectorizer(max_features=3000)
    X_cluster = cluster_vectorizer.fit_transform(df['cleaned_text'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['topic_id'] = kmeans.fit_predict(X_cluster)

    joblib.dump(cluster_vectorizer, cluster_vec_path)
    joblib.dump(kmeans, kmeans_path)
    return df

def assign_topics(df, cluster_vec_path, kmeans_path):
    if not os.path.exists(cluster_vec_path) or not os.path.exists(kmeans_path):
        logging.error("No trained cluster model found! Please train first.")
        return df

    df = df.copy()
    cluster_vectorizer = joblib.load(cluster_vec_path)
    kmeans = joblib.load(kmeans_path)
    X = cluster_vectorizer.transform(df['cleaned_text'])
    df['topic_id'] = kmeans.predict(X)
    return df

# -------------------------------------------------------------------------
# LABEL TOPIC TRENDS
# -------------------------------------------------------------------------
def label_topic_trends(df, date_col='date', topic_col='topic_id',
                       coverage_1day=10,
                       coverage_7day=15,
                       coverage_14day=19):
    df = df.copy()
    df = df.sort_values(by=[topic_col, date_col])

    coverage_map = {}
    grouped = df.groupby(topic_col)[date_col]
    for t, dates_series in grouped:
        unique_dates = sorted(list(set(dates_series.dropna().dt.date)))
        coverage_map[t] = unique_dates

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

        # SHIFT=1 => coverage in last 10 days
        dates_1d = [(cur_date - timedelta(days=i)) for i in range(10, -1, -1)]
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
# DATASET
# -------------------------------------------------------------------------
class TrendDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row_x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        return row_x, torch.tensor(self.y[idx], dtype=torch.long)

# -------------------------------------------------------------------------
# CLASSIFIER
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
# TRAINING LOOP
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

            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# -------------------------------------------------------------------------
# TRAIN + (OPTIONALLY) INFERENCE -> STORES TOP-5
# -------------------------------------------------------------------------
def train_for_country(country_code, do_train=True):
    """
    1) Load or fetch data for given country_code => data/{country_code}_data.csv
    2) If do_train, cluster + label + train SHIFT=1/7/14 models
    3) Then do an inference pass on the last 7 days to get top-5 articles
       for each SHIFT -> store to data/{country_code}_top5.csv
    """
    os.makedirs("data", exist_ok=True)
    gdelt_file = f"data/{country_code}_data.csv"
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    # ---------------------- LOAD or FETCH -----------------------------
    if os.path.exists(gdelt_file):
        logging.info(f"[{country_code}] Loading cached data from {gdelt_file}")
        n_data = pd.read_csv(gdelt_file)
        n_data['date'] = pd.to_datetime(n_data['date'], errors='coerce')
    else:
        articles = gdelt_fetch_articles_noquery_4weeks(country_code)
        if not articles:
            logging.error(f"[{country_code}] No GDELT data returned. Exiting.")
            return

        all_records = []
        for art in articles:
            pub_str = art.get('seendate')
            try:
                pub_date = datetime.strptime(pub_str, '%Y%m%dT%H%M%SZ').date() if pub_str else None
            except ValueError:
                pub_date = None
            all_records.append({
                'country': country_code,
                'date': pub_date,
                'title': art.get('title', ''),
                'description': art.get('extrasummary', '') or art.get('snippet', ''),
                'source': art.get('domain', ''),
                'url': art.get('url', '')
            })

        n_data = pd.DataFrame(all_records)
        n_data.drop_duplicates(subset=['url'], inplace=True)
        n_data.to_csv(gdelt_file, index=False)
        logging.info(f"[{country_code}] Fetched data saved to {gdelt_file}")

    # ---------------------- PREPROCESS -----------------------------
    n_data = preprocess_news(n_data)
    logging.info(f"[{country_code}] After preprocess, shape = {n_data.shape}")

    cutoff_date = end_date - timedelta(days=6)
    train_data = n_data[n_data['date'] < cutoff_date].copy()
    inference_data = n_data[n_data['date'] >= cutoff_date].copy()

    logging.info(f"[{country_code}] Train data shape={train_data.shape}; Inference shape={inference_data.shape}")

    # ---------------------- TRAIN -----------------------------
    if do_train:
        if train_data.empty:
            logging.error(f"[{country_code}] No data for training subset. Exiting.")
            return

        # 1) Cluster
        cluster_vec_path = f"data/cluster_vectorizer_{country_code}.pkl"
        kmeans_path = f"data/kmeans_model_{country_code}.pkl"
        train_data = train_cluster_model(train_data, 180, cluster_vec_path, kmeans_path)

        # 2) Label
        labeled_train = label_topic_trends(train_data)

        # 3) Train SHIFT=1,7,14
        for shift_day in [1,7,14]:
            label_col = f"label_shift_{shift_day}"
            vec_path = f"data/clf_vectorizer_{shift_day}days_{country_code}.pkl"
            model_path = f"data/trend_predictor_{shift_day}days_{country_code}.pth"

            vec = TfidfVectorizer(max_features=3000)
            X_text = vec.fit_transform(labeled_train['cleaned_text'])
            joblib.dump(vec, vec_path)

            y_final = labeled_train[label_col]
            c_dist = y_final.value_counts().to_dict()
            c0 = c_dist.get(0, 0)
            c1 = c_dist.get(1, 0)
            if c0 == 0 or c1 == 0:
                w = 1.0
            else:
                w = float(c0) / float(c1)
            w_tensor = torch.tensor([1.0, w], dtype=torch.float32)

            ds = TrendDataset(X_text, y_final)
            loader = DataLoader(ds, batch_size=32, shuffle=True)

            model = TrendPredictor(input_size=X_text.shape[1], hidden_size=128, num_classes=2)
            crit = nn.CrossEntropyLoss(weight=w_tensor)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)

            print(f"\n[{country_code}] Training SHIFT={shift_day} classifier. Distribution: {c_dist}")
            train_classifier(model, loader, crit, opt, epochs=10)
            torch.save(model.state_dict(), model_path)

        logging.info(f"[{country_code}] Finished training SHIFT=1,7,14 models.")

    # ---------------------- INFERENCE (last 7 days) + TOP 5 STORAGE -----------------------------
    if not inference_data.empty:
        # 1) Assign topics (KMeans)
        cluster_vec_path = f"data/cluster_vectorizer_{country_code}.pkl"
        kmeans_path = f"data/kmeans_model_{country_code}.pkl"
        inference_data = assign_topics(inference_data, cluster_vec_path, kmeans_path)
        if 'topic_id' not in inference_data.columns:
            logging.warning(f"[{country_code}] Could not assign topics. Skipping top-5.")
            return

        combined_infer = inference_data.copy()

        # 2) Predict SHIFT=1,7,14
        for shift_day in [1,7,14]:
            vec_path = f"data/clf_vectorizer_{shift_day}days_{country_code}.pkl"
            model_path = f"data/trend_predictor_{shift_day}days_{country_code}.pth"

            if not os.path.exists(vec_path) or not os.path.exists(model_path):
                logging.warning(f"[{country_code}] Missing SHIFT={shift_day} model. Skipping.")
                continue

            vec = joblib.load(vec_path)
            X_infer_text = vec.transform(combined_infer['cleaned_text'])

            model_infer = TrendPredictor(input_size=X_infer_text.shape[1], hidden_size=128, num_classes=2)
            st = torch.load(model_path)
            model_infer.load_state_dict(st)
            model_infer.eval()

            feats = torch.tensor(X_infer_text.toarray(), dtype=torch.float32)
            with torch.no_grad():
                out = model_infer(feats)
                probs = torch.softmax(out, dim=1)[:, 1].numpy()
                _, pred = torch.max(out, dim=1)

            pred_col = f'prediction_{shift_day}days'
            prob_col = f'prob_{shift_day}days'
            combined_infer[pred_col] = pred.numpy()
            combined_infer[prob_col] = probs

        # 3) Extract top-5 for each SHIFT and store in CSV
        all_top5 = []
        for shift_day in [1,7,14]:
            pred_col = f'prediction_{shift_day}days'
            prob_col = f'prob_{shift_day}days'
            if pred_col not in combined_infer.columns:
                continue

            trending = combined_infer[combined_infer[pred_col] == 1]
            if trending.empty:
                continue
            trending_sorted = trending.sort_values(by=prob_col, ascending=False)
            top5 = trending_sorted.head(5)

            for _, row in top5.iterrows():
                all_top5.append({
                    'shift': shift_day,
                    'date': row['date'],
                    'title': row['title'],
                    'prob': row[prob_col],
                    'url': row['url']
                })

        if all_top5:
            top5_df = pd.DataFrame(all_top5)
            out_file = f"data/{country_code}_top5.csv"
            top5_df.to_csv(out_file, index=False)
            logging.info(f"[{country_code}] Stored top-5 predictions to {out_file}")
        else:
            logging.info(f"[{country_code}] No articles predicted as trending. No top-5 stored.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Train TrendPredictor for a given country code, then store top-5 predictions.'
    )
    parser.add_argument('--country', type=str, required=True,
                        help='2-letter country code (e.g. US, JA)')
    parser.add_argument('--no-train', action='store_true',
                        help='Fetch data but skip training (still do top-5 inference if models exist)')
    args = parser.parse_args()

    train_for_country(args.country, do_train=not args.no_train)
