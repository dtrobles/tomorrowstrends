from pytrends.request import TrendReq
import pandas as pd

# Initialize PyTrends
pytrends = TrendReq(hl='en-US', tz=360)

# Define search term and time frame
search_term = "Python programming"
timeframe = "now 7-d"  # Last 7 days


# Get interest over time
data = pytrends.realtime_trending_searches(pn='US')

# Check if 'isPartial' column exists and drop it
if 'isPartial' in data.columns:
    data = data.drop(columns=['isPartial'])

# Display results
print(data)
