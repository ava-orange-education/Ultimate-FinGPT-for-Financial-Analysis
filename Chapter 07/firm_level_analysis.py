import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import datetime

# --- Configuration ---
STOCK_SYMBOLS = ["QSOFT.L", "NEX.L"]  # QuantumSoft Technologies (LSE), NexGen Robotics (LSE) - Adjust if needed
YAHOO_FINANCE_TIMEFRAME = "1y"  # Fetch 1 year of historical data
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your API key
FRED_SERIES_IDS = ["GBRGDPMYMEI", "GBRCPIALLMINMEI", "INTGSTHGB"]  # UK GDP, CPI, Interest Rate
NEWS_SOURCES = ["https://uk.finance.yahoo.com/", "https://www.reuters.com/finance/"]
SENTIMENT_API_URL = "YOUR_SENTIMENT_API_URL"  # Replace with your sentiment API endpoint
SENTIMENT_API_KEY = "YOUR_SENTIMENT_API_KEY"  # Replace with your sentiment API key
TWITTER_BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"  # Replace with your Twitter Bearer Token (if needed)
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"  # Replace with your Reddit Client ID (if needed)
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"  # Replace with your Reddit Client Secret (if needed)
REDDIT_USERNAME = "YOUR_REDDIT_USERNAME"  # Replace with your Reddit Username (if needed)
REDDIT_PASSWORD = "YOUR_REDDIT_PASSWORD"  # Replace with your Reddit Password (if needed)
SAVE_DATA_PATH = "data"
os.makedirs(SAVE_DATA_PATH, exist_ok=True)

# --- Utility Functions ---

def save_data(df, filename):
    """Saves a Pandas DataFrame to a CSV file."""
    filepath = os.path.join(SAVE_DATA_PATH, filename)
    df.to_csv(filepath)
    print(f"Data saved to: {filepath}")

def fetch_json_data(url, headers=None, params=None):
    """Fetches JSON data from a URL with optional headers and parameters."""
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

#  Historical Stock Prices (Yahoo Finance) 
def get_stock_prices(symbols, timeframe):
    """Fetches historical stock prices from Yahoo Finance."""
    data = yf.download(symbols, period=timeframe)
    return data

stock_prices_df = get_stock_prices(STOCK_SYMBOLS, YAHOO_FINANCE_TIMEFRAME)
if not stock_prices_df.empty:
    save_data(stock_prices_df, "stock_prices.csv")

#  Company Earnings Reports & Financial Statements (orinally the Requires Specific APIs or Web Scraping) ---
# Directly fetching and parsing these requires more complex logic and potentially specific APIs
# or web scraping of SEC filings or financial data providers. This is a placeholder.

def get_financial_statements(symbols):
    """Placeholder for fetching financial statements."""
    print("Fetching financial statements (placeholder - requires specific API or scraping).")
    financial_data = {}
    for symbol in symbols:
        # In a real scenario, you would use an API or scraping here
        financial_data[symbol] = {
            "balance_sheet": pd.DataFrame(),
            "income_statement": pd.DataFrame(),
            "cash_flow_statement": pd.DataFrame()
            
        }
    
    return financial_data

financial_data = get_financial_statements(STOCK_SYMBOLS)



