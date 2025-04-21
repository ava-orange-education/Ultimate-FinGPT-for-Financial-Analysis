import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Parameters
companies = ['QuantumSoft Technologies', 'NexGen Robotics']
start_date = datetime(2024, 1, 1)
days = 100

data = []
events = []

# Generate company-level stock data and events
for company in companies:
    price = 100 + np.random.randn() * 10
    for day in range(days):
        date = start_date + timedelta(days=day)
        open_price = price + np.random.randn()
        close_price = open_price + np.random.randn() * 2
        high = max(open_price, close_price) + np.random.rand() * 2
        low = min(open_price, close_price) - np.random.rand() * 2
        volume = random.randint(50000, 200000)
        earnings = np.random.uniform(0.1, 3.0)
        sentiment = np.random.uniform(-1, 1)

        data.append({
            'date': date,
            'company': company,
            'open': round(open_price, 2),
            'close': round(close_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'volume': volume,
            'earnings': round(earnings, 2),
            'sentiment': round(sentiment, 2),
        })

        # Inject synthetic news events
        if company == 'NexGen Robotics' and day == 60:
            events.append({
                'date': date,
                'company': company,
                'headline': 'NexGen Robotics wins $1.2B government defense contract',
                'event_type': 'positive',
                'sentiment_score': 0.9
            })
        elif company == 'QuantumSoft Technologies' and day == 45:
            events.append({
                'date': date,
                'company': company,
                'headline': 'TechCloud Services faces regulatory investigation; industry exposure increases',
                'event_type': 'negative',
                'sentiment_score': -0.75
            })

# Generate synthetic FRED-style macroeconomic indicators
# FRED = Federal Reserve Economic Data
fred_indicators = [
    {'indicator': 'Federal Funds Rate (%)', 'base': 5.0, 'volatility': 0.1},
    {'indicator': 'CPI Inflation Rate (%)', 'base': 3.2, 'volatility': 0.2},
    {'indicator': 'Unemployment Rate (%)', 'base': 4.0, 'volatility': 0.1},
    {'indicator': 'Industrial Production Index', 'base': 105.0, 'volatility': 0.5},
    {'indicator': 'Consumer Sentiment Index', 'base': 85.0, 'volatility': 2.0}
]

fred_data = []

for i in range(days):
    date = start_date + timedelta(days=i)
    for item in fred_indicators:
        value = round(np.random.normal(loc=item['base'], scale=item['volatility']), 2)
        fred_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'indicator': item['indicator'],
            'value': value
        })

# Convert to DataFrames
data_df = pd.DataFrame(data)
events_df = pd.DataFrame(events)
fred_df = pd.DataFrame(fred_data)

# Save separate files for clarity
data_df.to_csv('synthetic_stock_data.csv', index=False)
events_df.to_csv('synthetic_news_events.csv', index=False)
fred_df.to_csv('synthetic_fred_data.csv', index=False)

print("Synthetic company and FRED data generated.")
