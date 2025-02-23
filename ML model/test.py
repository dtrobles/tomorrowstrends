from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Initialize the driver and open the Trends page
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://trends.google.com/trending?geo=US&hours=168")

# Wait for the table (tbody) to load
table_selector = "#trend-table > div.enOdEe-wZVHld-zg7Cn-haAclf > table > tbody:nth-child(3)"
table_element = WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
)

# Wait until at least one row (<tr>) appears in the table
WebDriverWait(driver, 20).until(
    lambda d: len(d.find_elements(By.CSS_SELECTOR, table_selector + " tr")) > 0
)

# Get all <tr> elements from the table
rows = table_element.find_elements(By.TAG_NAME, "tr")

# Extract trend names (assumed to be in the first <div> within the second <td> of each row)
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

# Output the list of trend names
print("Trend names:", trend_names)

driver.quit()

# -------------------------------
# Now use pytrends to get the popularity (interest over time)
# -------------------------------
from pytrends.request import TrendReq
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
# Initialize pytrends
pytrend = TrendReq(hl='en-US', tz=360)

# Prepare an empty DataFrame to collect the popularity data
popularity_df = pd.DataFrame()

# Google Trends allows up to 5 keywords per payload.
# Process the trend_names in chunks of 5.
for i in range(0, len(trend_names), 5):
    chunk = trend_names[i:i+5]
    pytrend.build_payload(chunk, timeframe='now 1-d', geo='US')
    df = pytrend.interest_over_time()
    
    if not df.empty:
        # Remove the "isPartial" column if it exists.
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        # Select only the columns for the current chunk of terms
        popularity_df = pd.concat([popularity_df, df[chunk]], axis=1)

# Display the combined popularity data
print("Popularity data over the past 24 hours:")
print(popularity_df)
