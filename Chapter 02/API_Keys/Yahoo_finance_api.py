
# To run this code, you need to install the yfinance library. You can install it using pip
# pip install yfinance
# on the terminal type,  python yahoo_finance_api.py historical AAPL DATE_FROM  DATE_TO

import sys
import json
import yfinance as yf

def get_stock_info(symbol):
    """Fetch stock information."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return json.dumps(info, indent=4)
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_historical_data(symbol, start_date, end_date):
    """Fetch historical stock data."""
    try:
        stock = yf.Ticker(symbol)
        historical = stock.history(start=start_date, end=end_date)
        return historical.to_json()
    except Exception as e:
        return json.dumps({"error": str(e)})

def main():
    """Command-line interface for the Yahoo Finance API."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python yahoo_finance_api.py stock <symbol>")
        print("  python yahoo_finance_api.py historical <symbol> <start_date> <end_date>")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "stock" and len(sys.argv) == 3:
        symbol = sys.argv[2]
        print(get_stock_info(symbol))
    elif command == "historical" and len(sys.argv) == 5:
        symbol = sys.argv[2]
        start_date = sys.argv[3]
        end_date = sys.argv[4]
        print(get_historical_data(symbol, start_date, end_date))
    else:
        print("Invalid command or arguments.")
        print("Usage:")
        print("  python yahoo_finance_api.py stock <symbol>")
        print("  python yahoo_finance_api.py historical <symbol> <start_date> <end_date>")

if __name__ == "__main__":
    main()
