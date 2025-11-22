import asyncio
import pandas as pd
import os
from app.config import settings
from app.fetch_data import fetch_latest_prices
from app.model_runner import predict

RAW_DATA_PATH = "data/raw_data.csv"

async def fetch_loop():
    print("🔄 Scheduler started (Automatic Daily Prediction Mode)...")

    # --- MAIN LOOP ---
    while True:
        print("📡 Checking for new daily data...")
        new_data = fetch_latest_prices()

        if new_data:
            # 1. Load existing CSV to check for duplicates
            if os.path.exists(RAW_DATA_PATH):
                df_existing = pd.read_csv(RAW_DATA_PATH)
                existing_records = set(zip(df_existing["symbol"], df_existing["time"]))
            else:
                df_existing = pd.DataFrame()
                existing_records = set()

            rows_to_add = []
            
            for sym, val in new_data.items():
                record_key = (sym, val["time"])
                
                if record_key not in existing_records:
                    rows_to_add.append({
                        "symbol": sym,
                        "price": val["price"],
                        "volume": val["volume"],
                        "time": val["time"]
                    })
                    print(f"🆕 New candle found for {sym} at {val['time']}")
                
            # 2. Save only new rows
            if rows_to_add:
                df_new = pd.DataFrame(rows_to_add)
                df_new.to_csv(RAW_DATA_PATH, mode="a", header=not os.path.exists(RAW_DATA_PATH), index=False)
                print(f"✅ Saved {len(df_new)} new rows.")
            else:
                print("💤 No new unique data to save. Using last known data for forecast.")


        # --- CRITICAL CHANGE: ALWAYS PREDICT ---
        # Whether new data was added or not, we run the prediction pipeline.
        # This ensures the forecast is refreshed every 24 hours.
        print("🔮 Triggering prediction pipeline...")
        # Note: If models/scalers are missing, predict() will print a warning and return.
        predict() 


        # Wait for next interval (24 hours)
        print(f"⏳ Sleeping for {settings.fetch_interval} seconds...")
        await asyncio.sleep(settings.fetch_interval)

if __name__ == "__main__":
    asyncio.run(fetch_loop())