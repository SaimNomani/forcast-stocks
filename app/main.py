import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))

from app.config import settings
from app.db_manager import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic for FastAPI.
    """
    # Startup: Initialize database
    print("🚀 Starting FastAPI application...")
    db.connect()
    db.create_tables()
    print("✅ Database initialized")
    
    yield
    
    # Shutdown: Close database connection
    print("🔌 Shutting down FastAPI application...")
    db.close()


app = FastAPI(
    title="PSX Forecaster API",
    description="Stock price forecasting API with PostgreSQL backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "PSX Forecaster API is running!",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/config")
def get_config():
    """Get current configuration."""
    return {
        "global_stocks": settings.global_stocks,
        "local_stocks": settings.local_stocks,
        "fetch_interval_seconds": settings.fetch_interval_seconds,
        "local_update_interval_hours": settings.local_update_interval_hours
    }


@app.get("/forecast")
def get_latest_forecast(symbol: Optional[str] = None):
    """
    Get the latest forecast for a specific symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'OGDC.KA')
    
    Returns:
        Latest prediction data
    """
    if not symbol:
        raise HTTPException(status_code=400, detail="Please provide a symbol parameter")

    # Check database connection
    if not db.conn or db.conn.closed:
        db.connect()
    
    if not db.conn:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    # Determine table based on symbol
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
                raise HTTPException(
                    status_code=404, 
                    detail=f"No forecast found for {symbol}"
                )
                
            return {
                "symbol": row[0],
                "created_at": row[1].isoformat() if row[1] else None,
                "predicted_price": float(row[2]) if row[2] else None,
                "forecast_json": row[3],
                "explanation": row[4]
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/history")
def get_history(symbol: Optional[str] = None, limit: int = 60):
    """
    Get historical actual price data for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'OGDC.KA')
        limit: Number of records to return (default: 60)
    
    Returns:
        Historical price data
    """
    if not symbol:
        raise HTTPException(status_code=400, detail="Please provide a symbol parameter")

    # Validate limit
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")

    # Check database connection
    if not db.conn or db.conn.closed:
        db.connect()
    
    if not db.conn:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    # Determine table
    is_global = symbol in settings.global_stocks
    table = "global_stock_prices" if is_global else "local_stock_prices"
    
    query = f"""
        SELECT created_at, close_price, volume 
        FROM {table} 
        WHERE symbol = %s 
        ORDER BY created_at DESC 
        LIMIT %s
    """
    
    try:
        with db.conn.cursor() as cur:
            cur.execute(query, (symbol, limit))
            rows = cur.fetchall()
            
            result = []
            for row in rows:
                result.append({
                    "time": row[0].isoformat() if row[0] else None,
                    "price": float(row[1]) if row[1] else None,
                    "volume": int(row[2]) if row[2] else 0
                })
            
            return {
                "symbol": symbol,
                "count": len(result),
                "history": result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/stocks")
def get_all_stocks():
    """Get list of all available stocks."""
    return {
        "global_stocks": settings.global_stocks,
        "local_stocks": settings.local_stocks,
        "total": len(settings.global_stocks) + len(settings.local_stocks)
    }


@app.get("/health")
def health_check():
    """Detailed health check endpoint."""
    # Check database connection
    db_status = "healthy"
    try:
        if not db.conn or db.conn.closed:
            db.connect()
        if not db.conn:
            db_status = "unhealthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }


if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port=8000)