from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
import pandas as pd
from app.config import settings
from app.db_manager import db
from app.fetch_data import fetch_latest_prices
from app.prediction_pipeline import run_prediction_pipeline

def process_stock(symbol: str, data: dict, is_global: bool):
    """
    Process a single stock: save to DB and run prediction.
    """
    price = data["price"]
    volume = data["volume"]
    
    # 1. Save to DB
    db.insert_price(symbol, price, volume, is_global)
    
    # 2. Run Prediction Pipeline
    # We need history for the pipeline. 
    # For now, we might need to fetch more history if not in DB, 
    # but the pipeline expects a DataFrame.
    # Let's assume we fetch a small history from YFinance for the pipeline 
    # if we don't have a full local history yet.
    # OR we can just fetch the last 60 days from YFinance right here to pass to the pipeline.
    # This is safer to ensure we have data.
    
    try:
        # Fetch history for prediction (need at least 60+14 days)
        # We fetch 100 days to be safe.
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo", interval="1d") # 6 months is plenty
        
        if hist.empty:
            print(f"⚠️ No history found for {symbol}, skipping prediction.")
            return

        # Prepare DataFrame for pipeline
        df_history = hist.reset_index()
        df_history = df_history.rename(columns={"Close": "close_price", "Volume": "volume"})
        
        # Run Pipeline
        prediction_result, error = run_prediction_pipeline(symbol, df_history, is_global)
        
        if prediction_result:
            db.insert_prediction(
                symbol, 
                prediction_result["predicted_price"], 
                prediction_result["forecast_json"], 
                prediction_result["explanation_text"], 
                is_global
            )
        else:
            print(f"⚠️ Prediction failed for {symbol}: {error}")
            
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")

def scheduled_job():
    print(f"\n⏰ Scheduler Job Started at {datetime.now()}")
    
    # --- 1. Global Stocks (Always Run) ---
    print("🌍 Processing Global Stocks...")
    global_data = fetch_latest_prices(settings.global_stocks)
    
    for symbol, data in global_data.items():
        process_stock(symbol, data, is_global=True)
        
    # --- 2. Local Stocks (Check 24h) ---
    print("🇵🇰 Processing Local Stocks...")
    local_data = fetch_latest_prices(settings.local_stocks)
    
    for symbol, data in local_data.items():
        last_update = db.get_last_update_time(symbol, is_global=False)
        
        should_update = False
        if not last_update:
            should_update = True
            print(f"🆕 First time update for {symbol}")
        else:
            # Check if 24 hours passed
            time_diff = datetime.now() - last_update
            if time_diff > timedelta(hours=24):
                should_update = True
                print(f"🕒 >24h since last update for {symbol} (Last: {last_update})")
            else:
                print(f"zzz Skipping {symbol} (Updated {time_diff} ago)")
        
        if should_update:
            process_stock(symbol, data, is_global=False)

    print("✅ Job Finished.\n")

def start_scheduler():
    scheduler = BlockingScheduler()
    # Run every 15 minutes
    scheduler.add_job(scheduled_job, 'interval', minutes=15)
    
    print("🚀 Scheduler Service Started (Running every 15 mins)...")
    
    # Run once immediately on startup
    scheduled_job()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Scheduler stopped.")

if __name__ == "__main__":
    # Ensure tables exist
    db.create_tables()
    start_scheduler()
