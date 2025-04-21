import pandas as pd

import numpy as np

# Load synthetic data
stock_df = pd.read_csv("Data/synthetic_stock_data.csv")
news_df = pd.read_csv("Data/synthetic_news_events.csv")
fred_df = pd.read_csv("Data/synthetic_fred_data.csv")

#  Firm-level analysis
# Example: Compute rolling average and volatility for each company
stock_df['date'] = pd.to_datetime(stock_df['date'])
stock_df = stock_df.sort_values(by=['company', 'date'])

# Rolling metrics
stock_df['rolling_mean_close'] = stock_df.groupby('company')['close'].transform(lambda x: x.rolling(window=5).mean())
stock_df['rolling_volatility'] = stock_df.groupby('company')['close'].transform(lambda x: x.rolling(window=5).std())


#  Macro-level analysis
# Convert FRED data to wide format
fred_df['date'] = pd.to_datetime(fred_df['date'])
fred_wide = fred_df.pivot(index='date', columns='indicator', values='value').reset_index()
fred_wide.columns.name = None

# Example: Add macro regime label
conditions = [
    fred_wide['Federal Funds Rate (%)'] > 5.5,
    fred_wide['CPI Inflation Rate (%)'] > 4.0,
]
choices = ['High Rate Environment', 'High Inflation Environment']
fred_wide['macro_regime'] = np.select(conditions, choices, default='Stable')


# Merge stock + news
news_df['date'] = pd.to_datetime(news_df['date'])
combined_firm = pd.merge(news_df, stock_df, on=['date', 'company'], how='left')

# Merge in macro data
final_combined = pd.merge(combined_firm, fred_wide, on='date', how='left')


