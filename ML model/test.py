from pytrends.request import TrendReq
import pandas as pd

# Initialize Pytrends
pytrend = TrendReq(hl='en-US', tz=360)

# Define the keyword list and timeframe for the last 10 hours
kw_list = ['pasta']
timeframe = 'now 10-H'  # "now 10-H" indicates data from the past 10 hours

# Build the payload for the keyword and timeframe
pytrend.build_payload(kw_list=kw_list, timeframe=timeframe)

# Retrieve the interest over time data
data = pytrend.interest_over_time()

# Optionally drop the 'isPartial' column if you don't need it
if 'isPartial' in data.columns:
    data = data.drop(columns=['isPartial'])

# Display the results
print(data)
