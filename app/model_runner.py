from datetime import datetime, timedelta
import pandas as pd
import os
from app.model_utils import load_forecast_model, load_scaler, forecast_next_n_prices
from app.config import settings

PRED_PATH = "logs/predictions.csv"
RAW_PATH = "data/raw_data.csv"
LOOKBACK_WINDOW = 60 # Must match what you train with in Colab

def save_prediction_results(df_forecast: pd.DataFrame):
    """
    Saves prediction DataFrame, ensuring header is written only if file does not exist.
    """
    file_exists = os.path.exists(PRED_PATH)
    
    # Use 'a' (append) mode
    # Set header=True only if the file is new (False if appending)
    df_forecast.to_csv(PRED_PATH, 
                        mode="a", 
                        header=not file_exists, 
                        index=False)
    print(f"✅ Saved {len(df_forecast)} predictions to {PRED_PATH}")


def predict(n_steps=10):
    if not os.path.exists(RAW_PATH):
        print("⚠️ No raw data found.")
        return {}

    # Load raw data
    df = pd.read_csv(RAW_PATH, parse_dates=["time"])
    
    # Load the Model (Loaded once)
    model = load_forecast_model()
    if not model:
        print("⚠️ Model not found in models/ directory. Skipping prediction.")
        return {"error": "Model not found"}

    results = {}
    forecast_entries = []
    now = datetime.utcnow()

    # Loop through EACH symbol defined in config
    for symbol in settings.symbols:
        print(f"🔮 Predicting for {symbol}...")
        
        # 1. Get Data for this symbol
        df_symbol = df[df["symbol"] == symbol].sort_values(by="time")
        
        if len(df_symbol) < LOOKBACK_WINDOW:
            print(f"⚠️ Not enough data for {symbol} (Has {len(df_symbol)}, needs {LOOKBACK_WINDOW})")
            continue

        # 2. Load Scaler for this symbol
        scaler = load_scaler(symbol)
        if not scaler:
            print(f"⚠️ No scaler found for {symbol}. Skipping.")
            continue

        # 3. Prepare Input Data
        recent_closes = df_symbol["price"].values[-LOOKBACK_WINDOW:]

        try:
            # 4. Generate Forecast
            predictions = forecast_next_n_prices(
                model, 
                scaler, 
                recent_closes, 
                n_steps=n_steps, 
                window_size=LOOKBACK_WINDOW
            )

            # 5. Format Results
            symbol_forecasts = []
            for i, pred in enumerate(predictions):
                # For EOD, we add Days
                forecast_time = now + timedelta(days=(i + 1))
                
                entry = {
                    "symbol": symbol,
                    "time": forecast_time.isoformat(),
                    "predicted": round(float(pred), 2),
                    "actual": None
                }
                symbol_forecasts.append(entry)
                forecast_entries.append(entry)
            
            results[symbol] = symbol_forecasts

        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")

    # 6. Save All Predictions to CSV using the dedicated function
    if forecast_entries:
        df_forecast = pd.DataFrame(forecast_entries)
        save_prediction_results(df_forecast)
    
    return results