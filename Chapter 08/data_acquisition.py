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
price_volume_data.to_csv('Data/price_volume_data.csv')


#  Blockchain Transactions
wallet_types = ["whale", "retail", "exchange"]
blockchain_data = pd.DataFrame({
    "timestamp": random_dates(start_date, end_date, 100),
    "token": random.choices(tokens, k=100),
    "wallet_type": random.choices(wallet_types, k=100),
    "tx_volume_usd": np.round(np.random.uniform(1000, 10000000, 100), 2)
})
blockchain_data.to_csv('Data/blockchain_data.csv')


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
news_data.to_csv('Data/news_data.csv')


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
social_data.to_csv('Data/social_data.csv')


#  Project Announcements & Tech Updates
categories = ["Upgrade", "Security", "Funding", "Launch"]
announcements_data = pd.DataFrame({
    "date": random_dates(start_date, end_date, 100),
    "project_name": random.choices(tokens, k=100),
    "announcement_text": ["Update on project " + str(i) for i in range(100)],
    "category": random.choices(categories, k=100),
    "source_url": ["https://example.com/update" + str(i) for i in range(100)]
})

announcements_data.to_csv('Data/announcements_data.csv')


## Get synthetic dataset for sentiment labelled crypto statemments
import random

def generate_sentiment_labeled_crypto_data(num_samples=50):
    
    crypto_names = ["Bitcoin", "Ethereum", "Solana", "Cardano", "Dogecoin", "Shiba Inu", "BNB", "XRP", "Avalanche", "Polkadot"]
    positive_keywords = ["surge", "rally", "bullish", "pump", "gains", "increase", "growth", "promising", "strong", "upward trend", "to the moon", "breakout", "adoption", "integration"]
    negative_keywords = ["crash", "dump", "bearish", "plunge", "losses", "decrease", "drop", "risky", "uncertainty", "downward pressure", "correction", "sell-off", "regulation fears", "scam"]
    neutral_keywords = ["sideways", "stable", "flat", "consolidating", "trading", "market watch", "current price", "analysis", "report", "updates", "potential", "considering", "monitoring"]

    statements = []
    for _ in range(num_samples):
        crypto = random.choice(crypto_names)
        sentiment_choice = random.choices(['positive', 'neutral', 'negative'], weights=[0.35, 0.3, 0.35], k=1)[0]  # Slightly more weight to extremes

        if sentiment_choice == 'positive':
            keyword = random.choice(positive_keywords)
            statement = f"{crypto} is experiencing a significant {keyword}."
            if random.random() < 0.4:
                statement += f" Analysts predict further {random.choice(['gains', 'growth'])}."
            elif random.random() < 0.3:
                statement += f" The market sentiment around {crypto} is very {random.choice(['optimistic', 'positive'])}."
        elif sentiment_choice == 'negative':
            keyword = random.choice(negative_keywords)
            statement = f"There's a major {keyword} in the price of {crypto}."
            if random.random() < 0.4:
                statement += f" Investors are expressing {random.choice(['fear', 'concern'])} about the future."
            elif random.random() < 0.3:
                statement += f" Regulatory news is putting {random.choice(['strong', 'significant'])} {random.choice(['downward pressure', 'uncertainty'])} on {crypto}."
        else:
            keyword = random.choice(neutral_keywords)
            statement = f"The price of {crypto} is currently {keyword}."
            if random.random() < 0.4:
                statement += f" Traders are waiting for the next market catalyst."
            elif random.random() < 0.3:
                statement += f" Market analysis shows {crypto} in a period of {random.choice(['consolidation', 'stability'])}."

        statements.append({'text': statement, 'label': sentiment_choice})

    return statements

# Generate the dataset
crypto_sentiment_data = generate_sentiment_labeled_crypto_data(50)

# save as a dataset along with the instruction
# Convert to DataFrame
data = pd.DataFrame(crypto_sentiment_data)


data['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
data.columns = ['input','output','instruction']
print(data.head())

# save data

data.to_csv('Data/crypto_sentiment_data.csv', index= False)