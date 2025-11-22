import pandas as pd
import yfinance as yf
import os
import sys

# Ensure we can access app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import fetch_historical_data function
from app.fetch_data import fetch_historical_data 
from app.config import settings

RAW_DATA_PATH = "data/raw_data.csv"

def run_initial_fetch():
    """Fetches 5 years of historical data for all stocks and saves it locally."""
    print("--- Running Initial Data Fetch (5 Years History) ---")
    
    # 1. Fetch data using the existing function
    df_hist = fetch_historical_data(period="5y")
    
    if df_hist.empty:
        print("❌ FAILED: Could not retrieve historical data for PSX stocks.")
        return

    # 2. Ensure data directory exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    # 3. Save the data
    df_hist.to_csv(RAW_DATA_PATH, index=False)
    
    print(f"✅ Success! Saved {len(df_hist)} rows to {RAW_DATA_PATH}")
    print("You now have the base data needed for the model to make predictions.")

if __name__ == "__main__":
    run_initial_fetch()