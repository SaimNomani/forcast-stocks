import asyncio
from fastapi import FastAPI, HTTPException
from app.config import settings
from app.scheduler_service import start_scheduler
from app.db_manager import db
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start scheduler
    # The scheduler in scheduler_service is blocking, so we might need to run it in a separate thread 
    # OR change it to AsyncIOScheduler if we want it in the same loop.
    # However, scheduler_service.start_scheduler() uses BlockingScheduler which blocks.
    # For a web app, we usually run the scheduler in a background thread or use AsyncIOScheduler.
    # Given the current implementation of scheduler_service, let's just initialize the DB here.
    # The scheduler might need to be run as a separate process or thread.
    # For simplicity in this refactor, we'll assume the user runs the scheduler separately 
    # OR we can start it in a thread.
    
    # Let's just ensure DB tables exist
    db.create_tables()
    yield
    # Shutdown logic if needed

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
    return {"message": "PSX Forecaster API is running (DB Backend)!"}

@app.get("/config")
def get_config():
    return {
        "global_stocks": settings.global_stocks,
        "local_stocks": settings.local_stocks,
        "interval": settings.fetch_interval_seconds
    }

@app.get("/forecast")
def get_latest_forecast(symbol: Optional[str] = None):
    """Returns the latest forecast for a symbol from the DB."""
    if not symbol:
        return {"error": "Please provide a symbol parameter."}

    # Determine table based on symbol (heuristic or check both)
    # We'll check both tables or assume based on config
    is_global = symbol in settings.global_stocks
    table = "global_predictions" if is_global else "local_predictions"
    
    query = f"""
        SELECT symbol, created_at, predicted_price, forecast_json, explanation_text 
        FROM {table} 
        WHERE symbol = %s 
        ORDER BY created_at DESC 
        LIMIT 1
    """
    
    try:
        with db.conn.cursor() as cur:
            cur.execute(query, (symbol,))
            row = cur.fetchone()
            
            if not row:
                return {"message": f"No forecast found for {symbol}"}
                
            return {
                "symbol": row[0],
                "created_at": row[1],
                "predicted_price": float(row[2]),
                "forecast_json": row[3],
                "explanation": row[4]
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history(symbol: Optional[str] = None):
    """Returns historical actual data from DB."""
    if not symbol:
        return {"error": "Please provide a symbol parameter."}

    is_global = symbol in settings.global_stocks
    table = "global_stock_prices" if is_global else "local_stock_prices"
    
    query = f"""
        SELECT created_at, close_price, volume 
        FROM {table} 
        WHERE symbol = %s 
        ORDER BY created_at DESC 
        LIMIT 60
    """
    
    try:
        with db.conn.cursor() as cur:
            cur.execute(query, (symbol,))
            rows = cur.fetchall()
            
            result = []
            for row in rows:
                result.append({
                    "time": row[0],
                    "price": float(row[1]),
                    "volume": row[2]
                })
            return {"history": result}
    except Exception as e:
        return {"error": str(e)}