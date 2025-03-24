import pandas as pd
import numpy as np
import requests
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Fetch Stock Data from Alpha Vantage API
API_KEY = 'your_alpha_vantage_api_key'  # Replace with your actual Alpha Vantage API key
symbol = 'AAPL'  # Example: Apple Inc. stock symbol
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

response = requests.get(url)
data = response.json()

# Convert the data to DataFrame
stock_data = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
stock_data = stock_data.astype(float)  # Convert to float for numerical operations

# Step 2: Clean Data
# Handling missing data by forward filling
stock_data.fillna(method='ffill', inplace=True)

# Step 3: Outlier Detection using Z-score method
stock_data['zscore'] = zscore(stock_data['4. close'])  # Z-score for 'close' prices
outliers = stock_data[stock_data['zscore'].abs() > 3]  # Values with z-score > 3
print("Outliers detected:")
print(outliers)

# Step 4: Stationarity Testing using ADF test
result = adfuller(stock_data['4. close'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If data is non-stationary, apply differencing
if result[1] > 0.05:
    print("Data is non-stationary, applying differencing.")
    stock_data['close_diff'] = stock_data['4. close'] - stock_data['4. close'].shift(1)
    stock_data.dropna(inplace=True)  # Drop missing values after differencing

# Step 5: Feature Engineering
# Create lag features (previous day's closing price)
stock_data['close_lag_1'] = stock_data['4. close'].shift(1)
stock_data['close_lag_2'] = stock_data['4. close'].shift(2)

# Create rolling mean (5-day moving average)
stock_data['5_day_avg'] = stock_data['4. close'].rolling(window=5).mean()

# Drop missing values caused by lag features and rolling mean
stock_data.dropna(inplace=True)

# Step 6: Scaling/Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
stock_data[['4. close', 'close_lag_1', 'close_lag_2', '5_day_avg']] = scaler.fit_transform(
    stock_data[['4. close', 'close_lag_1', 'close_lag_2', '5_day_avg']])

# Step 7: Train-Test Split (80% Train, 20% Test)
train_size = int(len(stock_data) * 0.8)
train, test = stock_data[:train_size], stock_data[train_size:]

# Prepare features (X) and target (y) for training
X_train = train[['close_lag_1', 'close_lag_2', '5_day_avg']]
y_train = train['4. close']
X_test = test[['close_lag_1', 'close_lag_2', '5_day_avg']]
y_test = test['4. close']

# Step 8: Visualizing the Data
plt.figure(figsize=(12,6))
plt.plot(stock_data.index, stock_data['4. close'], label='Stock Price (Closing)')
plt.title('Stock Price Time-Series Data')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Step 9: Optional - Save the Preprocessed Data to CSV (for further analysis or training)
stock_data.to_csv('preprocessed_stock_data.csv')


'''•	Outliers: If any data points are found to be outliers based on the Z-score, they are printed for review.
•	ADF Test Results: The Augmented Dickey-Fuller test results indicate whether the data is stationary. If not, the data is differenced to make it stationary.
•	Preprocessed Data: The cleaned and transformed stock price data is available for further use in model training and prediction.
Notes:
•	You can adjust the API key and stock symbol based on your requirements.
•	This code focuses on daily stock prices (TIME_SERIES_DAILY). You can modify it for other frequencies (e.g., TIME_SERIES_INTRADAY for hourly data).
•	Further modifications like adding more features, experimenting with other scaling methods, or applying more advanced transformations can be done based on the model requirements.
'''