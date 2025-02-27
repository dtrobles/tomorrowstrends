#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import argparse
from db import SessionLocal, Prediction

DATA_DIR = "data"

def get_country_data_folder(country_code):
    """
    Return the directory where this country's data and predictions are stored.
    """
    folder_path = os.path.join(DATA_DIR, country_code.upper())
    return folder_path

def load_predictions(country_code):
    session = SessionLocal()
    try:
        records = session.query(Prediction).filter(Prediction.country_code == country_code).all()
        preds = [{"horizon": r.horizon, "term": r.term} for r in records]
    finally:
        session.close()
    return preds

def main():
    parser = argparse.ArgumentParser(description="Fetch predictions for Tomorrow's Trends.")
    parser.add_argument("--country", required=True, help="Country code (e.g. US, GB, CA).")
    args = parser.parse_args()

    country_code = args.country.upper()
    print(f"[INFO] Loading predictions for country={country_code}...")

    preds = load_predictions(country_code)
    if not preds:
        print(f"[WARN] No predictions found for country {country_code}.")
        print(f"[INFO] Please run model.py with the appropriate arguments to generate predictions.")
        return

    print(f"[INFO] Found {len(preds)} prediction rows. Displaying by horizon...")
    horizons = ["tomorrow", "3_days", "5_days"]
    for horizon in horizons:
        terms_list = [p["term"] for p in preds if p["horizon"] == horizon]
        if not terms_list:
            print(f"[WARN] No predictions found for horizon='{horizon}' in country={country_code}.")
            continue
        print(f"\n[INFO] Top terms predicted for {horizon} (country={country_code}):")
        for term in terms_list:
            print(f"  - {term}")

if __name__ == "__main__":
    main()