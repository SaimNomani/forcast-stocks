import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def calculate_metrics(csv_path: str, window_size: int = 10):
    try:
        if not os.path.exists(csv_path):
            return {"error": "Predictions file not found."}
            
        df = pd.read_csv(csv_path)
        
        # Drop rows where actual is missing
        df = df.dropna(subset=["actual"])

        if df.empty:
            return {"message": "No evaluated predictions yet (waiting for actuals)."}

        metrics_per_stock = {}
        
        # Group by symbol to get per-stock metrics
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values(by="time").tail(window_size)
            
            if len(group) < 2:
                continue

            y_true = group["actual"]
            y_pred = group["predicted"]

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Handle MAPE division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                if not np.isfinite(mape):
                    mape = 0.0

            metrics_per_stock[symbol] = {
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "MAPE": round(mape, 2)
            }

        return metrics_per_stock

    except Exception as e:
        return {"error": str(e)}