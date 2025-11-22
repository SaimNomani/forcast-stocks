import asyncio
from fastapi import FastAPI
from app.config import settings
from app.scheduler import fetch_loop
from app.model_runner import predict
from app.evaluator import calculate_metrics
from contextlib import asynccontextmanager
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start scheduler loop
    task = asyncio.create_task(fetch_loop())
    yield
    # Shutdown: Cancel loop
    task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "PSX Forecaster API is running!"}

@app.get("/config")
def get_config():
    return {
        "symbols": settings.symbols,
        "interval": settings.fetch_interval
    }

@app.get("/predict")
def make_prediction():
    """Manually trigger prediction pipeline."""
    return predict()
    
@app.get("/metrics")
def get_metrics():
    return calculate_metrics("logs/predictions.csv", window_size=10)

@app.get("/forecast")
def get_latest_forecast(symbol: Optional[str] = None):
    """Returns last 10 forecasts. Optional filtering by symbol."""
    pred_path = "logs/predictions.csv"
    if not os.path.exists(pred_path):
        return {"message": "No forecast data available yet."}

    try:
        df = pd.read_csv(pred_path)
        if symbol:
            df = df[df["symbol"] == symbol]

        result = []
        # Get last 10 entries per symbol
        grouped = df.sort_values(by="time").groupby("symbol")
        for sym, group in grouped:
            if symbol and sym != symbol: continue
            
            last_10 = group.tail(10)
            for _, row in last_10.iterrows():
                result.append({
                    "symbol": row["symbol"],
                    "time": row["time"],
                    "predicted": float(row["predicted"]) if pd.notna(row["predicted"]) else None,
                    "actual": float(row["actual"]) if pd.notna(row.get("actual", None)) else None
                })
        return {"forecasts": result}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/actuals")
def get_actuals(symbol: Optional[str] = None):
    """Returns historical actual data."""
    path = "data/raw_data.csv"
    if not os.path.exists(path):
        return {"message": "No actual data yet."}

    try:
        df = pd.read_csv(path)
        if symbol:
            df = df[df["symbol"] == symbol]
        
        result = []
        # Return last 60 days per symbol
        grouped = df.sort_values(by="time").groupby("symbol")
        for sym, group in grouped:
            if symbol and sym != symbol: continue
            
            subset = group.tail(60)
            for _, row in subset.iterrows():
                result.append({
                    "symbol": sym,
                    "time": row["time"],
                    "actual": float(row["price"])
                })
        return {"actuals": result}
    except Exception as e:
        return {"error": str(e)}