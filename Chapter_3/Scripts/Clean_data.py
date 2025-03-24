import requests
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import hashlib
import difflib
from transformers import pipeline

# Ensure necessary NLTK components are available
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Initialize a contextual sentiment analysis model (e.g., BERT-based model trained for financial news)
contextual_analyzer = pipeline("sentiment-analysis", model="finbert/finbert-sentiment")

# Alpha Vantage API Configuration
API_KEY = 'YOUR API KEY'
BASE_URL = 'https://www.alphavantage.co/query'

# Define source credibility ranking
SOURCE_CREDIBILITY = {
    "Reuters": 5, "Bloomberg": 5, "CNBC": 4, "Financial Times": 4, "Forbes": 3, 
    "Yahoo Finance": 3, "Unknown": 1
}

# Step 1: Duplicate Content Detection
def remove_duplicates(df):
    df['summary_hash'] = df['summary'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    df.drop_duplicates(subset=['summary_hash'], keep='first', inplace=True)
    return df

# Step 2: Identify Near-Duplicates
def identify_near_duplicates(df, threshold=0.9):
    unique_articles = []
    for idx, row in df.iterrows():
        is_duplicate = False
        for article in unique_articles:
            similarity = difflib.SequenceMatcher(None, row['summary'], article['summary']).ratio()
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_articles.append(row)
    return pd.DataFrame(unique_articles)

# Step 3: Context Preservation During Cleaning
def preserve_context(text):
    context_terms = {
        'stock': 'equity',
        'bullish': 'positive market sentiment',
        'bearish': 'negative market sentiment'
    }
    for term, replacement in context_terms.items():
        text = text.replace(term, replacement)
    return text

# Step 4: Maintain Semantic Relationships
def maintain_semantics(text):
    semantic_terms = {
        'increase': 'rise',
        'growth': 'rise',
        'decrease': 'fall',
        'decline': 'fall'
    }
    for term, replacement in semantic_terms.items():
        text = text.replace(term, replacement)
    return text

# Step 5: Handling Ambiguous Content
def handle_ambiguity(text):
    ambiguous_terms = {
        'bullish': 'positive market sentiment',
        'bearish': 'negative market sentiment',
        'volatile': 'unstable market'
    }
    for term, replacement in ambiguous_terms.items():
        text = text.replace(term, replacement)
    return text

# Step 6: Extract Numerical Data
def extract_numerical_data(text):
    numerical_data = re.findall(r'\d+\.\d+%?|\$\d+(?:,\d{3})*(?:\.\d{1,2})?', text)
    return numerical_data

# Step 7: Sentiment Analysis and Final Processing (Handling Ambiguity and Context)
def contextual_sentiment_analysis(text):
    # Run the contextual analysis using a financial news-specific model
    result = contextual_analyzer(text)
    sentiment_label = result[0]['label']  # Positive, Negative, or Neutral
    confidence = result[0]['score']  # Confidence score
    
    return sentiment_label, confidence

# Step 8: Clean and Standardize Text
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'[^\w\s.%]', '', text)  # Remove special characters except percentages
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Step 9: Handle Missing Data
def impute_missing_data(df):
    df['published_date'].fillna('Unknown', inplace=True)  # Default for missing dates
    df['source'].fillna('Unknown', inplace=True)
    df['summary'].fillna('No Summary Available', inplace=True)
    return df

# Step 10: Bias Detection and Mitigation
def detect_bias(df):
    source_counts = Counter(df['source'])
    print("Source Distribution:", source_counts)
    return df  # Placeholder for rebalancing if needed

# Step 11: Preprocessing Pipeline for Financial News Data
def preprocess_news_data(news_data):
    articles = news_data.get('feed', [])
    cleaned_data = []
    
    for article in articles:
        title = clean_text(article.get('title', ''))
        summary = clean_text(article.get('summary', ''))
        source = article.get('source', 'Unknown')
        sentiment_score = sia.polarity_scores(summary)['compound']
        
        # Use contextual sentiment analysis to determine sentiment
        contextual_sentiment_label, confidence = contextual_sentiment_analysis(summary)
        
        # Choose sentiment from context or default to VADER-based sentiment
        verified_sentiment = contextual_sentiment_label if confidence > 0.7 else "Neutral"
        
        # Extract numerical data (e.g., stock prices, percentages)
        numerical_data = extract_numerical_data(summary)
        
        # Calculate credibility score based on the source
        credibility_score = SOURCE_CREDIBILITY.get(source, 2)
        
        # Step 3: Context Preservation
        summary = preserve_context(summary)
        summary = maintain_semantics(summary)  # Step 4: Maintain Semantic Relationships
        summary = handle_ambiguity(summary)  # Step 5: Handle Ambiguity
        
        cleaned_data.append({
            'title': title,
            'summary': summary,
            'source': source,
            'credibility_score': credibility_score,
            'published_date': article.get('time_published', 'Unknown'),
            'topics': ", ".join([topic.get('name', '') for topic in article.get('topics', []) if isinstance(topic, dict)]),
            'alpha_vantage_sentiment': article.get('overall_sentiment_label', 'Neutral'),
            'verified_sentiment': verified_sentiment,
            'confidence': confidence,
            'extracted_numerical_data': numerical_data
        })
    
    # Convert to DataFrame for further analysis
    df = pd.DataFrame(cleaned_data)
    
    # Remove duplicates and near-duplicates
    df = remove_duplicates(df)
    df = identify_near_duplicates(df)
    
    # Handle missing data and mitigate bias
    df = impute_missing_data(df)
    df = detect_bias(df)
    
    return df

# Step 12: Fetch Financial News from Alpha Vantage
def fetch_alpha_vantage_news():
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': API_KEY,
        'topics': 'technology, finance, energy',
        'sort': 'LATEST',
        'limit': 50
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

# Step 13: Fetch Financial News from Yahoo Finance
def fetch_yahoo_finance_news():
    yahoo_news_url = 'https://query1.finance.yahoo.com/v7/finance/news'
    params = {'category': 'technology, finance, energy', 'count': 50}
    response = requests.get(yahoo_news_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

# Main Execution: Fetch Data from Both Alpha Vantage and Yahoo Finance
alpha_vantage_data = fetch_alpha_vantage_news()
yahoo_finance_data = fetch_yahoo_finance_news()

# Combine both datasets
if alpha_vantage_data and yahoo_finance_data:
    combined_data = alpha_vantage_data['feed'] + yahoo_finance_data['items']
    cleaned_df = preprocess_news_data({'feed': combined_data})
    
    # Display the cleaned data
    print("Cleaned Data Sample:")
    print(cleaned_df.head())
    
    # Save the cleaned data to CSV
    cleaned_df.to_csv('cleaned_financial_news_combined.csv', index=False)
