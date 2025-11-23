import yfinance as yf
import pandas as pd
from datetime import datetime
from app.config import settings

def fetch_historical_data(period="5y"):
    """
    Fetches long-term history for initial database population.
    """
    symbols = settings.symbols
    all_data = []
    
    print(f"📜 Fetching {period} historical data for {symbols}...")
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="1d")
            
            for date, row in hist.iterrows():
                all_data.append({
                    "symbol": symbol,
                    "time": date.isoformat(),
                    "price": float(row["Close"]),
                    "volume": float(row["Volume"])
                })
        except Exception as e:
            print(f"❌ Error fetching history for {symbol}: {e}")
            
    return pd.DataFrame(all_data)

def fetch_latest_prices(symbols=None):
    """
    Fetches the single latest daily candle (EOD) for updates.
    If symbols is None, uses settings.symbols (legacy support).
    """
    if symbols is None:
        symbols = settings.symbols
        
    data = {}
    print(f"📡 Connecting to YFinance for: {symbols}")
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Fetch last 5 days to ensure we capture the most recent completed trading day
            hist = ticker.history(period="5d", interval="1d") 
            
            if not hist.empty:
                latest = hist.iloc[-1]
                data[symbol] = {
                    "price": float(latest["Close"]),
                    "volume": float(latest["Volume"]),
                    # Use the candle's actual date, not the current server time
                    "time": latest.name.isoformat() 
                }
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")

    return data