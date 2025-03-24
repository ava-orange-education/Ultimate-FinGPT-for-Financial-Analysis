import requests
import pandas as pd
import os

# Alpha Vantage API Configuration
API_KEY = os.getenv('ALPHA_VANTAGE_KEY')
BASE_URL = 'https://www.alphavantage.co/query'

# Fetch financial news data
def fetch_alpha_vantage_news():
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': API_KEY,
        'topics': 'technology',  # Change to relevant topic
        'sort': 'LATEST',
        'limit': 50  # Number of articles to fetch
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

# Clean and preprocess the data
def preprocess_news_data(news_data):
    articles = news_data.get('feed', [])
    cleaned_data = []
    for article in articles:
        cleaned_data.append({
            'title': article.get('title'),
            'summary': article.get('summary'),
            'source': article.get('source'),
            'published_date': article.get('time_published'),
            'topics': article.get('topics', []),
            'sentiment': article.get('overall_sentiment_label')
        })
    return pd.DataFrame(cleaned_data)

# Main Execution
if __name__=='__main__':
    news_data = fetch_alpha_vantage_news()
    if news_data:
        print("Original Data:")
        print(news_data)
        cleaned_df = preprocess_news_data(news_data)
        print("Cleaned Data:")
        print(cleaned_df.head())
        # Save to CSV
        cleaned_df.to_csv('Data/alpha_vantage_news1.csv', index=False)
