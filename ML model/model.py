#!/usr/bin/env python3

import argparse
import os
import sys
import datetime
import pandas as pd
import numpy as np
import pickle
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
import time
import warnings

###############################################################################
# Constants / Config
###############################################################################
DATA_DIR = "data"  # main data directory
TOP_N_TERMS_PER_DAY = 20
LOOKBACK_DAYS = 7
pd.set_option('future.no_silent_downcasting', True)

###############################################################################
# Utility Functions
###############################################################################

def get_country_data_folder(country_code):
    """
    Return the directory where this country's data, model, predictions are stored.
    """
    folder_path = os.path.join(DATA_DIR, country_code.upper())
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def load_existing_data(country_code):
    """
    Load previously stored raw data if available, otherwise return an empty DataFrame.
    """
    folder = get_country_data_folder(country_code)
    data_path = os.path.join(folder, "raw_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=["date", "term", "value"])

def save_data(df, country_code):
    """
    Save the raw data DataFrame as CSV.
    """
    folder = get_country_data_folder(country_code)
    data_path = os.path.join(folder, "raw_data.csv")
    df.to_csv(data_path, index=False)

def save_model(model_dict, country_code):
    """
    Serialize the trained models (for tomorrow, 3-day, 5-day) to disk.
    """
    folder = get_country_data_folder(country_code)
    model_path = os.path.join(folder, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_dict, f)

def load_model(country_code):
    """
    Load previously saved models, if any.
    """
    folder = get_country_data_folder(country_code)
    model_path = os.path.join(folder, "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

def save_predictions(predictions_df, country_code):
    """
    Save the final predictions (for tomorrow, 3 days out, 5 days out).
    """
    folder = get_country_data_folder(country_code)
    preds_path = os.path.join(folder, "predictions.csv")
    predictions_df.to_csv(preds_path, index=False)

###############################################################################
# Trending Terms Fetching (using Selenium from file 2)
###############################################################################

def fetch_trending_terms_today(country_code):
    """
    Use Selenium to scrape the top trending searches from Google Trends.
    The URL is constructed with the provided country code.
    
    NOTE: This logic is based on the approach in file 2.
    """
    # Build the URL – note that file2 uses geo=US; here we generalize using the provided country code.
    url = f"https://trends.google.com/trending?geo={country_code.upper()}&hours=168"
    
    # Import Selenium-related modules
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager

    # Set up Chrome in headless mode
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(url)
    
    # CSS selector from file 2 for the trends table
    table_selector = "#trend-table > div.enOdEe-wZVHld-zg7Cn-haAclf > table > tbody:nth-child(3)"
    
    try:
        # Wait for the table to load
        table_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
        )
        
        # Wait until at least one row appears in the table
        WebDriverWait(driver, 20).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, table_selector + " tr")) > 0
        )
        
        # Get all rows in the table
        rows = table_element.find_elements(By.TAG_NAME, "tr")
        trend_names = []
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 2:
                continue
            details_td = cells[1]
            divs = details_td.find_elements(By.TAG_NAME, "div")
            if divs:
                name = divs[0].text.strip()
                if name:
                    trend_names.append(name)
        
        # Select only the top N trending terms
        terms = trend_names[:TOP_N_TERMS_PER_DAY]
    except Exception as e:
        print("Error fetching trends using Selenium:", e)
        terms = []
    finally:
        driver.quit()
    
    return terms

pd.options.mode.chained_assignment = None  # or other warnings filtering

###############################################################################
# Data Gathering & Cleaning
###############################################################################

def gather_data_for_past_seven_days(country_code):
    """
    1) Fetch today's top trending searches using Selenium.
    2) For each trending term, fetch the *hourly* interest for the past 7 days.
    3) Aggregate hourly data by date (daily), returning a DataFrame with
       columns: [date, term, value].
    """
    pytrends = TrendReq(hl='en-US', tz=360, retries=5, backoff_factor=0.1)
    
    top_terms = fetch_trending_terms_today(country_code)
    all_records = []
    for term in top_terms:
        # Avoid hammering the server:
        time.sleep(2)
        
        pytrends.build_payload([term], timeframe='now 7-d', geo='')
        hourly_data = pytrends.interest_over_time()
        
        if hourly_data.empty:
            continue

        # Drop isPartial if it exists
        if 'isPartial' in hourly_data.columns:
            hourly_data.drop(columns=['isPartial'], inplace=True)
        
        # Convert hourly timestamps to dates and aggregate by day
        hourly_data.reset_index(inplace=True)
        hourly_data['date'] = hourly_data['date'].dt.date
        
        daily_data = (
            hourly_data
            .groupby('date')[term]
            .mean()  # average normalized popularity over the day
            .reset_index(name='value')
        )
        
        daily_data['term'] = term
        daily_data = daily_data[['date', 'term', 'value']]
        all_records.append(daily_data)
    
    if not all_records:
        return pd.DataFrame(columns=["date", "term", "value"])
    
    df = pd.concat(all_records, ignore_index=True)
    return df

###############################################################################
# Model Training (Improved: separate labels for tomorrow, 3-days, 5-days)
###############################################################################

def build_features_labels(df):
    """
    Given a DataFrame of [date, term, value] for the past 7 days (daily-aggregated),
    build features that capture:
      - Daily popularity for each of the last 7 days
      - Slope of change in popularity
    And create three separate labels for tomorrow, 3 days, and 5 days.
    """
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['date'] = df['date'].dt.date
    
    # Keep only data within the last 7 days
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=LOOKBACK_DAYS)).date()
    df = df[df['date'] >= cutoff_date]
    
    # Pivot table: index=term, columns=date, values=value
    pivot_df = df.pivot_table(index='term', columns='date', values='value', aggfunc='mean').fillna(0)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    if pivot_df.shape[1] < 2:
        pivot_df['label_tomorrow'] = 0
        pivot_df['label_3day'] = 0
        pivot_df['label_5day'] = 0
        return pivot_df.dropna(axis=1), None, pivot_df.index, []
    
    dates_sorted = list(pivot_df.columns)
    last_day = dates_sorted[-1]
    second_last_day = dates_sorted[-2]
    pivot_df['slope_last2days'] = pivot_df[last_day] - pivot_df[second_last_day]
    
    slope_threshold = pivot_df['slope_last2days'].quantile(0.6)
    last_day_threshold = pivot_df[last_day].quantile(0.6)
    pivot_df['label_tomorrow'] = (
        (pivot_df['slope_last2days'] >= slope_threshold) |
        (pivot_df[last_day] >= last_day_threshold)
    ).astype(int)
    
    if pivot_df.shape[1] >= 3:
        last_3_days = dates_sorted[-3:]
        pivot_df['avg_last3'] = pivot_df[last_3_days].mean(axis=1)
    else:
        pivot_df['avg_last3'] = pivot_df[last_day]
    
    thr_3day = pivot_df['avg_last3'].quantile(0.7)
    pivot_df['label_3day'] = (pivot_df['avg_last3'] >= thr_3day).astype(int)
    
    if pivot_df.shape[1] >= 5:
        last_5_days = dates_sorted[-5:]
        pivot_df['avg_last5'] = pivot_df[last_5_days].mean(axis=1)
    else:
        pivot_df['avg_last5'] = pivot_df[dates_sorted].mean(axis=1)
    
    thr_5day = pivot_df['avg_last5'].quantile(0.8)
    pivot_df['label_5day'] = (pivot_df['avg_last5'] >= thr_5day).astype(int)
    
    label_cols = ['label_tomorrow', 'label_3day', 'label_5day']
    exclude_cols = label_cols + ['avg_last3', 'avg_last5']
    feature_cols = [c for c in pivot_df.columns if c not in exclude_cols]
    
    X = pivot_df[feature_cols].values
    y_tomorrow = pivot_df['label_tomorrow'].values
    y_3day = pivot_df['label_3day'].values
    y_5day = pivot_df['label_5day'].values
    terms_index = pivot_df.index
    return X, (y_tomorrow, y_3day, y_5day), terms_index, feature_cols

def train_model(df):
    """
    Train three separate RandomForestClassifiers for predicting trending terms
    for tomorrow, 3-day, and 5-day horizons.
    """
    X, (y_tomorrow, y_3day, y_5day), terms_index, feature_cols = build_features_labels(df)
    
    if X is None or len(terms_index) == 0 or X.shape[0] == 0:
        print("Not enough data to train a proper model. Returning None.")
        return None
    
    model_dict = {}
    
    if len(np.unique(y_tomorrow)) > 1:
        clf_tomorrow = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_tomorrow.fit(X, y_tomorrow)
        model_dict['tomorrow'] = clf_tomorrow
    else:
        model_dict['tomorrow'] = None
    
    if len(np.unique(y_3day)) > 1:
        clf_3day = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf_3day.fit(X, y_3day)
        model_dict['3_days'] = clf_3day
    else:
        model_dict['3_days'] = None
    
    if len(np.unique(y_5day)) > 1:
        clf_5day = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
        clf_5day.fit(X, y_5day)
        model_dict['5_days'] = clf_5day
    else:
        model_dict['5_days'] = None
    
    return model_dict

###############################################################################
# Predicting Future Trends
###############################################################################

def generate_predictions(model_dict, df):
    """
    Generate separate predictions for tomorrow, 3-day, and 5-day horizons.
    For each horizon, select the top 10 terms based on predicted probability,
    or fallback to a slope-based heuristic if no model is available.
    """
    X, (y_tomorrow, y_3day, y_5day), terms_index, feature_cols = build_features_labels(df)
    
    if X is None or X.shape[0] == 0:
        return pd.DataFrame(columns=["horizon", "term"])
    
    pivot_predictions = []
    
    def get_top_terms(probabilities, terms_index, top_n=10):
        idx_sorted = np.argsort(probabilities)[::-1]
        idx_top = idx_sorted[:top_n]
        return terms_index[idx_top].tolist()
    
    for horizon_key in ['tomorrow', '3_days', '5_days']:
        clf = model_dict.get(horizon_key, None)
        
        if clf is None:
            # Fallback: rank by slope_last2days
            fallback_df = df.copy()
            fallback_df['date'] = pd.to_datetime(fallback_df['date'])
            pivot_df = fallback_df.pivot_table(index='term', columns='date', values='value', aggfunc='mean').fillna(0)
            pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
            
            if pivot_df.shape[1] >= 2:
                dates_sorted = pivot_df.columns
                last_day = dates_sorted[-1]
                second_last_day = dates_sorted[-2]
                pivot_df['slope'] = pivot_df[last_day] - pivot_df[second_last_day]
            else:
                pivot_df['slope'] = 0.0
            
            top_terms = pivot_df['slope'].sort_values(ascending=False).head(10).index.tolist()
        else:
            probabilities = clf.predict_proba(X)[:, 1]
            top_terms = get_top_terms(probabilities, terms_index, top_n=10)
        
        for t in top_terms:
            pivot_predictions.append({"horizon": horizon_key, "term": t})
    
    predictions_df = pd.DataFrame(pivot_predictions)
    return predictions_df

###############################################################################
# Update Logic
###############################################################################

def update_data(df, new_df):
    """
    1) Remove data older than 7 days.
    2) Append new data, avoiding duplicates.
    """
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=LOOKBACK_DAYS)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df['date'] = df['date'].dt.date
    df = df[df['date'] >= cutoff.date()]
    
    if new_df.empty:
        return df
    
    new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce', utc=True)
    new_df['date'] = new_df['date'].dt.date
    new_df = new_df[new_df['date'] >= cutoff.date()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        updated_df = pd.concat([df, new_df], ignore_index=True)
    
    updated_df.drop_duplicates(subset=["date", "term"], keep="last", inplace=True)
    return updated_df

###############################################################################
# Main CLI Handler
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Tomorrow's Trends Model Script")
    parser.add_argument("--country", required=True, help="Country code, e.g. US, CA, GB, etc.")
    parser.add_argument("--update", action="store_true", help="If set, update existing data & model.")
    args = parser.parse_args()
    
    country_code = args.country.upper()
    
    # 1) Load existing data
    existing_data = load_existing_data(country_code)
    
    if args.update:
        print(f"Updating data for country={country_code}...")
        new_data = gather_data_for_past_seven_days(country_code)
        updated_data = update_data(existing_data, new_data)
        save_data(updated_data, country_code)
        
        model_dict = load_model(country_code)
        model_dict = train_model(updated_data)
        if model_dict:
            save_model(model_dict, country_code)
        
        preds_df = generate_predictions(model_dict, updated_data)
        save_predictions(preds_df, country_code)
        print("Update complete. Predictions file refreshed.")
        
    else:
        print(f"Gathering new data for country={country_code}...")
        new_data = gather_data_for_past_seven_days(country_code)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data.drop_duplicates(subset=["date", "term"], keep="last", inplace=True)
        combined_data = update_data(pd.DataFrame(columns=["date", "term", "value"]), combined_data)
        save_data(combined_data, country_code)
        
        model_dict = train_model(combined_data)
        if model_dict:
            save_model(model_dict, country_code)
        
        preds_df = generate_predictions(model_dict, combined_data)
        save_predictions(preds_df, country_code)
        print("Model training complete. Predictions saved.")

if __name__ == "__main__":
    main()
