import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from app.config import settings

# Paths
MODELS_DIR = "models"
GLOBAL_SCALERS_DIR = os.path.join(MODELS_DIR, "global_scalers")
LOCAL_SCALERS_DIR = os.path.join(MODELS_DIR, "local_scalers")
GLOBAL_MODEL_PATH = os.path.join(MODELS_DIR, "global_model.h5")
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "local_model.h5")

# Cache models to avoid reloading every time
_global_model = None
_local_model = None

def get_model(is_global: bool):
    """Load and return the appropriate model."""
    global _global_model, _local_model
    
    if is_global:
        if _global_model is None:
            if os.path.exists(GLOBAL_MODEL_PATH):
                _global_model = load_model(GLOBAL_MODEL_PATH)
                print("✅ Global model loaded.")
            else:
                print(f"❌ Global model not found at {GLOBAL_MODEL_PATH}")
        return _global_model
    else:
        if _local_model is None:
            if os.path.exists(LOCAL_MODEL_PATH):
                _local_model = load_model(LOCAL_MODEL_PATH)
                print("✅ Local model loaded.")
            else:
                print(f"❌ Local model not found at {LOCAL_MODEL_PATH}")
        return _local_model

def get_scaler(symbol: str, is_global: bool):
    """Load the specific scaler for the symbol."""
    # Replace dots with underscores for filename
    safe_symbol = symbol.replace(".", "_")
    scaler_filename = f"{safe_symbol}_scaler.pkl"
    
    directory = GLOBAL_SCALERS_DIR if is_global else LOCAL_SCALERS_DIR
    scaler_path = os.path.join(directory, scaler_filename)
    
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            # print(f"✅ Scaler loaded for {symbol}")
            return scaler
        except Exception as e:
            print(f"❌ Failed to load scaler for {symbol}: {e}")
            return None
    else:
        print(f"⚠️ Scaler not found for {symbol} at {scaler_path}")
        return None

def calculate_sma(series, length):
    return series.rolling(window=length).mean()

def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(df: pd.DataFrame, scaler):
    """Calculate indicators and scale data."""
    # Ensure we have enough data
    if len(df) < 60: # Assuming 60 is the window size
        return None, "Insufficient data"

    # Calculate Indicators Manually
    df['SMA_14'] = calculate_sma(df['close_price'], 14)
    df['RSI_14'] = calculate_rsi(df['close_price'], 14)
    
    # Drop NaN values created by indicators
    df.dropna(inplace=True)
    
    if len(df) < 60:
        return None, "Insufficient data after indicators"

    # Scale the 'close_price' (assuming model was trained on Close)
    # Note: If model was trained on multiple features, this needs adjustment.
    # Based on prompt, we just scale. Assuming scaler expects 2D array.
    scaled_data = scaler.transform(df[['close_price']].values)
    
    return scaled_data, None

def generate_explanation(last_price, sma_14, rsi_14):
    """Generate a simple rule-based explanation."""
    explanation = []
    
    if last_price > sma_14:
        explanation.append("Price is above SMA 14 (Bullish trend).")
    else:
        explanation.append("Price is below SMA 14 (Bearish trend).")
        
    if rsi_14 > 70:
        explanation.append("RSI is overbought (>70), potential pullback.")
    elif rsi_14 < 30:
        explanation.append("RSI is oversold (<30), potential bounce.")
    else:
        explanation.append(f"RSI is neutral ({round(rsi_14, 2)}).")
        
    return " ".join(explanation)

def run_prediction_pipeline(symbol: str, df_history: pd.DataFrame, is_global: bool):
    """
    Main function to run the prediction pipeline for a symbol.
    df_history should contain 'close_price' and 'volume'.
    """
    print(f"🚀 Starting prediction pipeline for {symbol}...")
    
    # 1. Load Resources
    model = get_model(is_global)
    if not model:
        return None, "Model missing"
        
    scaler = get_scaler(symbol, is_global)
    if not scaler:
        return None, "Scaler missing"

    # 2. Preprocessing
    # We need at least 60 + 14 (for RSI/SMA) data points ideally
    # But let's work with what we have.
    
    # Calculate indicators for explanation BEFORE scaling
    # (We need the latest values for explanation)
    df_calc = df_history.copy()
    df_calc['SMA_14'] = calculate_sma(df_calc['close_price'], 14)
    df_calc['RSI_14'] = calculate_rsi(df_calc['close_price'], 14)
    
    # Handle NaNs for explanation if needed, but we just take the last row
    # If last row is NaN, we can't explain properly.
    if df_calc.iloc[-1].isnull().any():
         # Try to fill or just warn?
         # If we have enough data, last row shouldn't be NaN unless data is bad.
         pass

    last_row = df_calc.iloc[-1]
    # Check if indicators are NaN
    if pd.isna(last_row['SMA_14']) or pd.isna(last_row['RSI_14']):
         explanation = "Not enough data for indicators."
    else:
        explanation = generate_explanation(last_row['close_price'], last_row['SMA_14'], last_row['RSI_14'])

    # Scale data for model
    # The model likely expects a sequence of length 60
    # We take the last 60 points
    scaled_data = scaler.transform(df_history[['close_price']].values)
    
    if len(scaled_data) < 60:
        return None, f"Not enough data ({len(scaled_data)} < 60)"
        
    input_seq = scaled_data[-60:].reshape(1, 60, 1) # (1, 60, 1)
    
    # 3. Prediction
    try:
        # Predict next 10 days (iterative or direct depending on model architecture)
        # Assuming model outputs 1 step ahead, we might need to loop.
        # OR if model outputs 10 steps at once. 
        # The prompt says "Predict next 10 days". 
        # Let's assume a simple loop for now if model output is shape (1,1)
        
        predictions = []
        current_batch = input_seq.copy()
        
        for _ in range(10):
            pred = model.predict(current_batch, verbose=0)[0] # Shape (1,) or (1,1)
            predictions.append(pred)
            
            # Update batch: remove first, add new pred
            # Reshape pred to (1, 1, 1) to match dimensions
            pred_reshaped = np.array(pred).reshape(1, 1, 1)
            current_batch = np.append(current_batch[:, 1:, :], pred_reshaped, axis=1)
            
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        inversed_predictions = scaler.inverse_transform(predictions)
        
        forecast_json = {
            f"day_{i+1}": round(float(val[0]), 2) 
            for i, val in enumerate(inversed_predictions)
        }
        
        predicted_price = float(inversed_predictions[0][0]) # Next day prediction
        
        return {
            "predicted_price": predicted_price,
            "forecast_json": forecast_json,
            "explanation_text": explanation
        }, None

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, str(e)
