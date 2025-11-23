import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db_manager import db
from app.prediction_pipeline import run_prediction_pipeline
from app.config import settings

def test_pipeline():
    print("🧪 Starting Pipeline Test...")

    # 1. Initialize DB
    print("🛠️ Initializing Database...")
    db.create_tables()

    # 2. Test Global Stock (AAPL)
    print("\n🍏 Testing Global Stock (AAPL)...")
    # Mock data for AAPL
    # We need at least 75 rows for indicators + 60 window
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    data = {
        "close_price": [150 + i * 0.1 for i in range(100)], # Linear uptrend
        "volume": [1000000 for _ in range(100)]
    }
    df_aapl = pd.DataFrame(data, index=dates)
    
    # Run pipeline
    # Note: This might fail if scaler/model is missing. 
    # We expect it to fail gracefully or succeed if files exist.
    result, error = run_prediction_pipeline("AAPL", df_aapl, is_global=True)
    
    if result:
        print("✅ AAPL Prediction Success!")
        print(f"Predicted Price: {result['predicted_price']}")
        print(f"Explanation: {result['explanation_text']}")
        
        # Save to DB
        db.insert_prediction("AAPL", result['predicted_price'], result['forecast_json'], result['explanation_text'], is_global=True)
    else:
        print(f"⚠️ AAPL Prediction Failed (Expected if model/scaler missing): {error}")

    # 3. Test Local Stock (HUBC.KA)
    print("\n🇵🇰 Testing Local Stock (HUBC.KA)...")
    df_hubc = pd.DataFrame(data, index=dates) # Reuse mock data
    
    result, error = run_prediction_pipeline("HUBC.KA", df_hubc, is_global=False)
    
    if result:
        print("✅ HUBC.KA Prediction Success!")
        print(f"Predicted Price: {result['predicted_price']}")
        
        # Save to DB
        db.insert_prediction("HUBC.KA", result['predicted_price'], result['forecast_json'], result['explanation_text'], is_global=False)
    else:
        print(f"⚠️ HUBC.KA Prediction Failed (Expected if model/scaler missing): {error}")

    print("\n✅ Test Complete.")

if __name__ == "__main__":
    test_pipeline()
