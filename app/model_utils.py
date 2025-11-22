import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# The model is shared (trained on all stocks), but scalers are specific
MODEL_PATH = "models/lstm_forecast_model.h5"
SCALER_DIR = "models"

def load_forecast_model():
    """Loads the shared LSTM model."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_scaler(symbol):
    """Loads the specific scaler for a stock (e.g., OGDC.KA_scaler.pkl)."""
    # Sanitize symbol name for filename (replace . with _)
    safe_symbol = symbol.replace(".", "_") 
    scaler_path = os.path.join(SCALER_DIR, f"{safe_symbol}_scaler.pkl")
    
    if not os.path.exists(scaler_path):
        print(f"⚠️ Scaler not found for {symbol} at {scaler_path}")
        return None
    
    return joblib.load(scaler_path)

def forecast_next_n_prices(model, scaler, close_prices: list, n_steps: int = 10, window_size: int = 60):
    """
    Generate N-step forecast using last closing prices.
    """
    # 1. Scale the data using the STOCK SPECIFIC scaler
    raw_array = np.array(close_prices).reshape(-1, 1)
    scaled_series = scaler.transform(raw_array)
    
    # 2. Check length
    if len(scaled_series) < window_size:
        raise ValueError(f"Not enough data. Need {window_size}, got {len(scaled_series)}")

    # 3. Prepare Input
    input_seq = scaled_series[-window_size:].reshape(1, window_size, 1)
    
    # 4. Predict
    predictions_scaled = model.predict(input_seq, verbose=0) 
    
    # 5. Inverse Transform
    forecasts = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    return forecasts