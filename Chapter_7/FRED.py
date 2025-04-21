import pandas as pd
import numpy as np
from datetime import timedelta, datetime
# Generate FRED-style macroeconomic indicators for the same date range

fred_indicators = [
    {'indicator': 'Federal Funds Rate (%)', 'base': 5.0, 'volatility': 0.1},
    {'indicator': 'CPI Inflation Rate (%)', 'base': 3.2, 'volatility': 0.2},
    {'indicator': 'Unemployment Rate (%)', 'base': 4.0, 'volatility': 0.1},
    {'indicator': 'Industrial Production Index', 'base': 105.0, 'volatility': 0.5},
    {'indicator': 'Consumer Sentiment Index', 'base': 85.0, 'volatility': 2.0}
]

fred_data = []

for i in range(100):  # Matching the first 100 days for simplicity
    date = datetime(2024, 1, 1) + timedelta(days=i)
    for item in fred_indicators:
        value = round(np.random.normal(loc=item['base'], scale=item['volatility']), 2)
        fred_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'indicator': item['indicator'],
            'value': value
        })

df_fred = pd.DataFrame(fred_data)

df_fred.to_csv('Data/fred_indicators.csv')