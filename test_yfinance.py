import yfinance as yf

symbol = "INFY"
ticker = f"{symbol}.NS"
data = yf.download(ticker, period="60d", interval="1d")

if data.empty:
    print("❌ No data fetched. Invalid ticker?")
else:
    print("✅ Data fetched successfully!")
    print(data.tail())
