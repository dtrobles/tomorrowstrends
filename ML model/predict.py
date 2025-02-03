#!/usr/bin/env python3
import argparse
import os
import pandas as pd

DATA_DIR = "data"

def get_country_data_folder(country_code):
    """
    Return the directory where this country's data and predictions are stored.
    """
    folder_path = os.path.join(DATA_DIR, country_code.upper())
    return folder_path

def load_predictions(country_code):
    """
    Load predictions from predictions.csv if available. Otherwise return an empty DataFrame.
    """
    folder = get_country_data_folder(country_code)
    preds_path = os.path.join(folder, "predictions.csv")
    if os.path.exists(preds_path):
        return pd.read_csv(preds_path)
    else:
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Fetch predictions for Tomorrow's Trends.")
    parser.add_argument("--country", required=True, help="Country code (e.g. US, GB, CA).")
    args = parser.parse_args()

    country_code = args.country.upper()
    print(f"[INFO] Loading predictions for country={country_code}...")

    preds_df = load_predictions(country_code)

    if preds_df.empty:
        print(f"[WARN] No predictions found for country {country_code}.")
        print("[INFO] Please run 'model.py --country {country}' first to generate predictions.")
        return

    print(f"[INFO] Found {len(preds_df)} prediction rows. Displaying by horizon...")

    # Display predictions for tomorrow, 3-days, and 5-days
    horizons = ["tomorrow", "3_days", "5_days"]
    for horizon in horizons:
        subset = preds_df[preds_df["horizon"] == horizon]
        if subset.empty:
            print(f"[WARN] No predictions found for horizon='{horizon}' in country={country_code}.")
            continue

        terms_list = subset["term"].tolist()
        print(f"\n[INFO] Top terms predicted for {horizon} (country={country_code}):")
        for term in terms_list:
            print(f"  - {term}")

if __name__ == "__main__":
    main()
