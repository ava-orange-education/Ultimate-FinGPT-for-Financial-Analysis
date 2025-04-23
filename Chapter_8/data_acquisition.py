import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Helper to generate random dates
def random_dates(start, end, n):
    return [start + timedelta(days=random.randint(0, (end - start).days)) for _ in range(n)]

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
tokens = ["BTC", "ETH", "SOL", "DOGE", "USDC"]

#  Historical Prices & Volumes
price_volume_data = pd.DataFrame({
    "date": random_dates(start_date, end_date, 100),
    "token": random.choices(tokens, k=100),
    "price_usd": np.round(np.random.uniform(10, 50000, 100), 2),
    "volume_usd": np.round(np.random.uniform(100000, 50000000, 100), 2)
})

#  Blockchain Transactions
wallet_types = ["whale", "retail", "exchange"]
blockchain_data = pd.DataFrame({
    "timestamp": random_dates(start_date, end_date, 100),
    "token": random.choices(tokens, k=100),
    "wallet_type": random.choices(wallet_types, k=100),
    "tx_volume_usd": np.round(np.random.uniform(1000, 10000000, 100), 2)
})

#  Financial News & Regulatory Alerts
news_sources = ["Bloomberg", "CoinDesk", "Reuters", "CryptoSlate"]
event_types = ["Regulation", "Market Shift", "Adoption", "ETF"]
news_data = pd.DataFrame({
    "date": random_dates(start_date, end_date, 100),
    "source": random.choices(news_sources, k=100),
    "headline": ["Headline " + str(i) for i in range(100)],
    "event_type": random.choices(event_types, k=100),
    "sentiment_score": np.round(np.random.uniform(-1, 1, 100), 2)
})

#  Social Media Sentiment
platforms = ["Twitter", "Reddit", "CryptoTalk"]
social_data = pd.DataFrame({
    "timestamp": random_dates(start_date, end_date, 100),
    "platform": random.choices(platforms, k=100),
    "token_mentioned": random.choices(tokens, k=100),
    "user_type": random.choices(["retail", "influencer"], k=100),
    "sentiment_score": np.round(np.random.uniform(-1, 1, 100), 2),
    "engagement": np.random.randint(1, 10000, 100)
})

#  Project Announcements & Tech Updates
categories = ["Upgrade", "Security", "Funding", "Launch"]
announcements_data = pd.DataFrame({
    "date": random_dates(start_date, end_date, 100),
    "project_name": random.choices(tokens, k=100),
    "announcement_text": ["Update on project " + str(i) for i in range(100)],
    "category": random.choices(categories, k=100),
    "source_url": ["https://example.com/update" + str(i) for i in range(100)]
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Synthetic Data Samples", dataframe=price_volume_data.head())
