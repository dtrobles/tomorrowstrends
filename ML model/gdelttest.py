#!/usr/bin/env python3

import requests
import time
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def fetch_gdelt_articles(query=None,
                        start_dt=None,
                        end_dt=None,
                        source_country=None,
                        maxrecords=250,
                        offset=0):
    """
    Quick function to pull data from GDELT's doc API and return raw JSON articles.
    - query: string (e.g. '"covid" sourcecountry:US' or 'sourcecountry:US')
    - start_dt/end_dt: YYYYMMDDHHMMSS string (e.g., '20230101000000')
    - source_country: optional, usually a 2-letter code
    - maxrecords: up to 250 (GDELT limit)
    - offset: for pagination (multiples of 250)
    """
    base_url = "http://api.gdeltproject.org/api/v2/doc/doc"
    
    # Build the query param
    # If user provided `query` they can embed sourcecountry in that query itself.
    # But if we want a separate param, we could do something like:
    # query_str = f'"{query}" sourcecountry:{source_country}' if (query and source_country) else query
    # For simplicity, we just use `query` directly below:
    params = {
            'query': f'"earthquake" sourcecountry:JA',
            'mode': 'artlist',
            'format': 'json',
            'maxrecords': '250',        # Only 250 articles
            'sort': 'DateDesc',
            'startdatetime': start_dt, # Start of this 7-day chunk
            'enddatetime': end_dt,     # End of this 7-day chunk
            'offset': 0                # No pagination
    }
    if start_dt:
        params['startdatetime'] = start_dt  # "YYYYMMDDHHMMSS"
    if end_dt:
        params['enddatetime'] = end_dt      # "YYYYMMDDHHMMSS"
    if offset:
        params['offset'] = offset

    headers = {
        'User-Agent': 'MyGdeltTestScript/1.0'
    }

    logging.info(f"Fetching GDELT with: {params}")

    try:
        resp = requests.get(base_url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        logging.info(f"Returned {len(articles)} articles.")
        return articles

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching from GDELT: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Test GDELT API parameters.")
    parser.add_argument("--query", default='sourcecountry:US', help="Query string to use. Example: '\"covid\" sourcecountry:US'")
    parser.add_argument("--start", default=None, help="Start datetime in YYYYMMDDHHMMSS format")
    parser.add_argument("--end", default=None, help="End datetime in YYYYMMDDHHMMSS format")
    parser.add_argument("--maxrecords", type=int, default=250, help="Max records (<=250) to fetch")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset (multiples of 250)")
    args = parser.parse_args()

    articles = fetch_gdelt_articles(query=args.query,
                                    start_dt=args.start,
                                    end_dt=args.end,
                                    maxrecords=args.maxrecords,
                                    offset=args.offset)
    time.sleep(1)  # Minimal sleep, you can increase if repeatedly calling in a loop

    # Print out some summary info
    print("\n=== SAMPLE OUTPUT ===")
    for i, art in enumerate(articles[:5], start=1):
        title = art.get("title", "NO_TITLE")
        url = art.get("url", "NO_URL")
        snippet = art.get("snippet", "")
        print(f"\nArticle #{i}")
        print(f"Title: {title}")
        print(f"URL: {url}")
        print(f"Snippet: {snippet[:100]}...")  # truncated

    print(f"\nTotal fetched articles: {len(articles)}")

if __name__ == "__main__":
    main()
