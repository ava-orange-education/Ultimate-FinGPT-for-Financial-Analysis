import random
import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime, timedelta

# Seed for consistency
random.seed(42)
np.random.seed(42)

# Companies
companies = ['QuantumSoft Technologies', 'NexGen Robotics']
start_date = datetime(2024, 1, 1)
entries_per_company = 100  # 100 per company for 200 total

# Event definitions for controlled sentiment
event_templates = {
    'positive': [
        "Strong quarterly results reported by {company}.",
        "{company} expands into international markets.",
        "Analysts upgrade {company} to 'Buy'."
    ],
    'negative': [
        "{company} faces data breach allegations.",
        "Weak demand in Q1 affects {company}'s performance.",
        "{company} stock downgraded amid supply chain issues."
    ],
    'neutral': [
        "{company} to hold annual investor meeting.",
        "{company} maintains previous guidance.",
        "No significant changes in {company}'s outlook."
    ],
    'absolutely_positive': [
        "{company} secures massive $1.2B government contract.",
        "Revolutionary AI tech by {company} receives global acclaim.",
        "{company} stock soars 20% on industry breakthrough."
    ],
    'absolutely_negative': [
        "{company} under federal investigation for accounting fraud.",
        "Major sell-off hits {company} after executive resignation.",
        "Market panic as {company} halts operations temporarily."
    ]
}

sources = [ 'Alpha Advantage',  'Twitter', 'Reddit']
sentiment_levels = list(event_templates.keys())

data = []

for company in companies:
    for i in range(entries_per_company):
        date = start_date + timedelta(days=i)

        # Choose source and sentiment
        source = random.choice(sources)
        sentiment = random.choices(
            sentiment_levels,
            weights=[2, 2, 2, 1, 1],  # more neutral/positive/negative, less extremes
            k=1
        )[0]

        text = random.choice(event_templates[sentiment]).format(company=company)

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'company': company,
            'source': source,
            'text': text,
            'sentiment': sentiment
        })

# Create DataFrame
df_synthetic_news = pd.DataFrame(data)


df_synthetic_news.to_csv('Data/news_sentiment.csv')