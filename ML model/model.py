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
# Data Gathering & Cleaning
###############################################################################

def fetch_trending_terms_today(country_code):
    """
    Use pytrends to fetch the top trending searches for the last 24 hours
    in the specified region/country.
    
    NOTE: `country_code` must map to a valid region code used by pytrends.
          For example: "US" -> "united_states", "GB" -> "united_kingdom", etc.
    """
    # Map ISO code to pytrends "pn" parameter. Adjust as needed:
    country_map = {
        "US": "united_states",
        "GB": "united_kingdom",
        "CA": "canada",
        "AU": "australia",
        "IN": "india",
        "JP": "japan",
        # Add more as needed...
    }
    region = country_map.get(country_code.upper(), "united_states")
    
    pytrends = TrendReq(hl='en-US', tz=360)
    df_trends = pytrends.trending_searches(pn=region)
    if df_trends.empty:
        return []
    
    terms = df_trends[0].head(TOP_N_TERMS_PER_DAY).tolist()
    return terms

pd.options.mode.chained_assignment = None  # or other warnings filtering


def gather_data_for_past_seven_days(country_code):
    """
    1) Fetch today's top 20 trending searches (once).
    2) For each trending term, fetch the *hourly* interest for the past 7 days.
    3) Aggregate hourly data by date (daily), returning a DataFrame with
       columns: [date, term, value].
    """
    pytrends = TrendReq(hl='en-US', tz=360, retries=5, backoff_factor=0.1)
    
    top_terms = fetch_trending_terms_today(country_code)
    all_records = []
    for term in top_terms:
        # Avoid hammering the server:
        time.sleep(2)  # Sleep 2 seconds before each request
        
        pytrends.build_payload([term], timeframe='now 7-d', geo='')
        hourly_data = pytrends.interest_over_time()
        
        if hourly_data.empty:
            continue

        
        # Drop isPartial if it exists
        if 'isPartial' in hourly_data.columns:
            hourly_data.drop(columns=['isPartial'], inplace=True)
        
        # 3) The index is hourly timestamps; convert to a column and aggregate by date
        hourly_data.reset_index(inplace=True)  # 'date' is now a column with hourly timestamps
        hourly_data['date'] = hourly_data['date'].dt.date  # keep only the date portion
        
        # Group by day (e.g., taking the mean or sum)
        daily_data = (
            hourly_data
            .groupby('date')[term]
            .mean()   # or .sum(), but mean is typical for normalized popularity
            .reset_index(name='value')
        )
        
        # Add the term column
        daily_data['term'] = term
        
        # Reorder columns to match your schema
        daily_data = daily_data[['date', 'term', 'value']]
        
        all_records.append(daily_data)
    
    # Combine data for all terms
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
      - Slopes or changes in popularity
    And create three separate labels:
      - label_tomorrow (predict if it will be 'trending' tomorrow)
      - label_3day (predict if it will be 'trending' 3 days from now)
      - label_5day (predict if it will be 'trending' 5 days from now)

    This is purely illustrative. Real logic might check actual "top 20" presence
    on subsequent days, etc.
    """
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['date'] = df['date'].dt.date
    
    # Keep only data within the last 7 days
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=LOOKBACK_DAYS)).date()
    df = df[df['date'] >= cutoff_date]
    
    # Create pivot: index=term, columns=date, values=value
    pivot_df = df.pivot_table(index='term', columns='date', values='value', aggfunc='mean').fillna(0)
    # Sort columns (ascending by date)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    # If we have fewer than 2 columns, just bail out
    if pivot_df.shape[1] < 2:
        pivot_df['label_tomorrow'] = 0
        pivot_df['label_3day'] = 0
        pivot_df['label_5day'] = 0
        return pivot_df.dropna(axis=1), None, pivot_df.index, []
    
    # Create simple features: the daily values themselves
    # Some might also add a slope feature: pivot_df[dayN] - pivot_df[dayN-1]
    # For illustration, let's add "last_day - second_last_day" as a slope
    dates_sorted = list(pivot_df.columns)
    last_day = dates_sorted[-1]
    second_last_day = dates_sorted[-2]
    
    pivot_df['slope_last2days'] = pivot_df[last_day] - pivot_df[second_last_day]
    
    # Example labels:
    # 1) label_tomorrow: 1 if there is a big slope from second_last_day -> last_day,
    #    or if the last_day's popularity is above 70th percentile
    slope_threshold = pivot_df['slope_last2days'].quantile(0.6)
    last_day_threshold = pivot_df[last_day].quantile(0.6)
    pivot_df['label_tomorrow'] = (
        (pivot_df['slope_last2days'] >= slope_threshold) |
        (pivot_df[last_day] >= last_day_threshold)
    ).astype(int)
    
    # 2) label_3day: 1 if the average of the last 3 days is above some threshold
    if pivot_df.shape[1] >= 3:
        last_3_days = dates_sorted[-3:]
        pivot_df['avg_last3'] = pivot_df[last_3_days].mean(axis=1)
    else:
        pivot_df['avg_last3'] = pivot_df[last_day]
    
    thr_3day = pivot_df['avg_last3'].quantile(0.7)
    pivot_df['label_3day'] = (pivot_df['avg_last3'] >= thr_3day).astype(int)
    
    # 3) label_5day: 1 if the average of the last 5 days is above some threshold
    if pivot_df.shape[1] >= 5:
        last_5_days = dates_sorted[-5:]
        pivot_df['avg_last5'] = pivot_df[last_5_days].mean(axis=1)
    else:
        pivot_df['avg_last5'] = pivot_df[dates_sorted].mean(axis=1)
    
    thr_5day = pivot_df['avg_last5'].quantile(0.8)
    pivot_df['label_5day'] = (pivot_df['avg_last5'] >= thr_5day).astype(int)
    
    # Build up feature matrix
    # We'll take all the daily columns + slope as features (excluding label columns).
    label_cols = ['label_tomorrow', 'label_3day', 'label_5day']
    exclude_cols = label_cols + ['avg_last3', 'avg_last5']
    # The daily columns + slope
    feature_cols = [c for c in pivot_df.columns if c not in exclude_cols]
    
    X = pivot_df[feature_cols].values
    y_tomorrow = pivot_df['label_tomorrow'].values
    y_3day = pivot_df['label_3day'].values
    y_5day = pivot_df['label_5day'].values
    terms_index = pivot_df.index
    return X, (y_tomorrow, y_3day, y_5day), terms_index, feature_cols

def train_model(df):
    """
    Train three separate RandomForestClassifiers (for tomorrow, 3-day, and 5-day).
    Return a dictionary of models keyed by horizon.
    """
    X, (y_tomorrow, y_3day, y_5day), terms_index, feature_cols = build_features_labels(df)
    
    # Edge case: if we got None or X is empty
    if X is None or len(terms_index) == 0 or X.shape[0] == 0:
        print("Not enough data to train a proper model. Returning None.")
        return None
    
    model_dict = {}
    
    # Train model for "tomorrow"
    if len(np.unique(y_tomorrow)) > 1:
        clf_tomorrow = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_tomorrow.fit(X, y_tomorrow)
        model_dict['tomorrow'] = clf_tomorrow
    else:
        model_dict['tomorrow'] = None
    
    # Train model for "3_days"
    if len(np.unique(y_3day)) > 1:
        clf_3day = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf_3day.fit(X, y_3day)
        model_dict['3_days'] = clf_3day
    else:
        model_dict['3_days'] = None
    
    # Train model for "5_days"
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
    We'll pick the top 10 terms (by predicted probability) for each horizon.
    If a particular model doesn't exist or is None, we fallback to a slope-based heuristic.
    """
    # Rebuild the same features
    X, (y_tomorrow, y_3day, y_5day), terms_index, feature_cols = build_features_labels(df)
    
    if X is None or X.shape[0] == 0:
        # fallback: no data
        return pd.DataFrame(columns=["horizon", "term"])
    
    pivot_predictions = []
    
    # Helper function
    def get_top_terms(probabilities, terms_index, top_n=10):
        # Sort descending
        idx_sorted = np.argsort(probabilities)[::-1]
        idx_top = idx_sorted[:top_n]
        return terms_index[idx_top].tolist()
    
    # For each horizon, produce top 10
    for horizon_key in ['tomorrow', '3_days', '5_days']:
        clf = model_dict.get(horizon_key, None)
        
        if clf is None:
            # fallback: rank by slope_last2days from pivot
            # Re-fetch pivot to see slope
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
            
            # top 10 by slope
            top_terms = pivot_df['slope'].sort_values(ascending=False).head(10).index.tolist()
        else:
            # predict_proba
            probabilities = clf.predict_proba(X)[:, 1]  # prob of label=1
            top_terms = get_top_terms(probabilities, terms_index, top_n=10)
        
        # Collect them
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
    2) Append new data, avoiding duplicates if needed.
    """
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=LOOKBACK_DAYS)
    
    # Safely convert to datetime (fix #2 below in the same code)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    # Keep only the date portion
    df['date'] = df['date'].dt.date
    df = df[df['date'] >= cutoff.date()]
    
    # If new_df is empty, just return df
    if new_df.empty:
        return df
    
    # Otherwise, safely convert new_df date as well
    new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce', utc=True)
    new_df['date'] = new_df['date'].dt.date
    new_df = new_df[new_df['date'] >= cutoff.date()]

    # Now concatenate without warning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        updated_df = pd.concat([df, new_df], ignore_index=True)

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
        # Gather brand-new data from the past 7 days (using today's top terms)
        new_data = gather_data_for_past_seven_days(country_code)
        # Merge it with existing
        updated_data = update_data(existing_data, new_data)
        # Save raw data
        save_data(updated_data, country_code)
        
        # Load or create model, re-train
        model_dict = load_model(country_code)
        model_dict = train_model(updated_data)
        if model_dict:
            save_model(model_dict, country_code)
        
        # Generate predictions
        preds_df = generate_predictions(model_dict, updated_data)
        save_predictions(preds_df, country_code)
        print("Update complete. Predictions file refreshed.")
        
    else:
        print(f"Gathering new data for country={country_code}...")
        # Fresh data
        new_data = gather_data_for_past_seven_days(country_code)
        
        # Combine with existing (if any)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data.drop_duplicates(subset=["date", "term"], keep="last", inplace=True)
        
        # Keep only last 7 days
        combined_data = update_data(pd.DataFrame(columns=["date", "term", "value"]), combined_data)
        
        # Save
        save_data(combined_data, country_code)
        
        # Train model
        model_dict = train_model(combined_data)
        if model_dict:
            save_model(model_dict, country_code)
        
        # Predict
        preds_df = generate_predictions(model_dict, combined_data)
        save_predictions(preds_df, country_code)
        print("Model training complete. Predictions saved.")

if __name__ == "__main__":
    main()