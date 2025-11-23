import pandas as pd
import os
import sys
from datetime import datetime

# Ensure we can access app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.fetch_data import fetch_historical_data 
from app.config import settings
from app.db_manager import db

def run_initial_fetch():
    """Fetches 5 years of historical data for all stocks and saves it to the DB."""
    print("--- Running Initial Data Fetch (5 Years History) ---")
    
    # 1. Fetch data
    df_hist = fetch_historical_data(period="5y")
    
    if df_hist.empty:
        print("❌ FAILED: Could not retrieve historical data.")
        return

    print(f"✅ Fetched {len(df_hist)} rows. Inserting into Database...")
    
    # 2. Insert into DB
    db.create_tables()
    
    count = 0
    for _, row in df_hist.iterrows():
        symbol = row['symbol']
        price = row['price']
        volume = row['volume']
        time_str = row['time'] # ISO format string
        
        # Convert time string to datetime
        try:
            created_at = datetime.fromisoformat(time_str)
        except ValueError:
            # Handle potential format issues
            created_at = pd.to_datetime(time_str).to_pydatetime()

        is_global = symbol in settings.global_stocks
        
        db.insert_price(symbol, price, volume, is_global, created_at=created_at)
        count += 1
        
        if count % 100 == 0:
            print(f"   Inserted {count} rows...", end="\r")
            
    print(f"\n✅ Success! Inserted {count} rows into the Database.")
    print("You now have the base data needed for the model and API history.")

if __name__ == "__main__":
    run_initial_fetch()