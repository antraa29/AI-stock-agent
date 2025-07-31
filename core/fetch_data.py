import yfinance as yf
import pandas_ta as ta
import pandas as pd

def get_stock_data(symbol, period='2y', interval='1d'):
    """
    Fetch stock data from Yahoo Finance.
    """
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)

    # Flatten multi-index columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.dropna(inplace=True)
    return data

def add_indicators(df):
    """
    Add technical indicators to the stock dataframe.
    """
    df = df.copy()

    # Core indicators
    df['RSI']    = ta.rsi(df['Close'], length=14)
    df['MACD']   = ta.macd(df['Close'])['MACD_12_26_9']
    df['EMA20']  = ta.ema(df['Close'], length=20)

    # Additional indicators for improved accuracy
    df['SMA50']  = ta.sma(df['Close'], length=50)
    df['ADX']    = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['ATR']    = ta.atr(df['High'], df['Low'], df['Close'])
    df['WILLR']  = ta.willr(df['High'], df['Low'], df['Close'])
    df['ROC']    = ta.roc(df['Close'])

    # Drop rows with missing values (common after indicators)
    df.dropna(inplace=True)

    # Debug preview
    print("\n--- Data Sample ---")
    print(df[['Close', 'RSI', 'MACD', 'EMA20']].head())
    print(f"NaNs in Close: {df['Close'].isna().sum()}")
    print(f"Total rows: {len(df)}")

    return df

if __name__ == "__main__":
    df = get_stock_data("INFY.NS")
    df = add_indicators(df)
    print("\n--- Final Data Preview ---")
    print(df.tail())
