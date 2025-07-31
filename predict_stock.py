import sys
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime

# === Load Model ===
try:
    model = joblib.load("models/xgb_model.pkl")
    print("✅ Model loaded successfully.\n")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit()

# === Get symbols from CLI ===
if len(sys.argv) < 2:
    print("⚠️  Usage: python predict_stock.py SYMBOL1 SYMBOL2 ...")
    print("ℹ️  Example: python predict_stock.py AAPL TSLA MSFT")
    exit()

symbols = [s.upper() for s in sys.argv[1:]]

# === Prediction Function ===
def get_prediction(symbol):
    try:
        print(f"\n🔍 Downloading data for: {symbol}")
        df = yf.download(symbol, period="3mo", interval="1d")

        if df.empty:
            raise ValueError("No data found.")

        # === Flatten multi-indexed columns if needed ===
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print("\n📊 Raw DataFrame head:")
        print(df.head())
        print("\n🧩 DataFrame columns:")
        print(df.columns)

        df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

        # === Calculate Technical Indicators ===
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd()
        df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['WILLR'] = WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        df['ROC'] = ROCIndicator(df['Close']).roc()

        # === Features for Prediction ===
        features = df[["RSI", "MACD", "EMA20", "SMA50", "ADX", "ATR", "WILLR", "ROC"]].dropna()

        if features.empty:
            raise ValueError("Not enough data for indicators.")

        latest_features = features.tail(1)
        prediction = model.predict(latest_features)[0]
        label = "Buy" if prediction == 1 else "Sell"

        return {
            "Symbol": symbol,
            "Date": df.index[-1].strftime('%Y-%m-%d'),
            "RSI": latest_features["RSI"].values[0],
            "MACD": latest_features["MACD"].values[0],
            "EMA20": latest_features["EMA20"].values[0],
            "SMA50": latest_features["SMA50"].values[0],
            "ADX": latest_features["ADX"].values[0],
            "ATR": latest_features["ATR"].values[0],
            "WILLR": latest_features["WILLR"].values[0],
            "ROC": latest_features["ROC"].values[0],
            "Prediction": label
        }

    except Exception as e:
        print(f"❌ Error while processing {symbol}: {e}")
        return {
            "Symbol": symbol,
            "Date": "N/A",
            "RSI": None,
            "MACD": None,
            "EMA20": None,
            "SMA50": None,
            "ADX": None,
            "ATR": None,
            "WILLR": None,
            "ROC": None,
            "Prediction": "Error"
        }

# === Predict all symbols and save ===
results = []
for symbol in symbols:
    result = get_prediction(symbol)
    results.append(result)
    print(f"📍 {symbol} | Prediction: {result['Prediction']}")

df_out = pd.DataFrame(results)
csv_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_out.to_csv(csv_filename, index=False)
print(f"\n💾 Saved predictions to {csv_filename}")
