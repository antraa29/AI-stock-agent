import yfinance as yf
import pandas as pd

def get_full_ticker(symbol):
    """Append .NS for known Indian stocks, else return as is."""
    indian_stocks = ['INFY', 'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'WIPRO', 'SBIN']  # add more if needed
    if symbol.upper() in indian_stocks:
        return f"{symbol.upper()}.NS"
    return symbol.upper()

def fetch_and_engineer_features(symbol, period='60d', interval='1d'):
    ticker = get_full_ticker(symbol)
    data = yf.download(ticker, period=period, interval=interval)

    if data.empty:
        raise ValueError("No data fetched. Check if the ticker symbol is correct.")

    # Feature engineering
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Close'].rolling(window=5).std()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Volume_Change'] = data['Volume'].pct_change()

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Optional label column for training
    data['Signal'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    return data[['Return', 'Volatility', 'MA5', 'MA10', 'MA20', 'Volume_Change', 'Signal']]
