import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))
from app.config import settings


class DBManager:
    def _init_(self):
        self.conn = None

    def connect(self):
        """Establish a connection to the database."""
        if self.conn and not self.conn.closed:
            return  # Already connected
            
        try:
            self.conn = psycopg2.connect(settings.database_url)
            self.conn.autocommit = True
            print("✅ Connected to Database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.conn = None

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        if not self.conn or self.conn.closed:
            self.connect()
        
        if not self.conn:
            print("❌ Cannot create tables - no database connection")
            return

        queries = [
            """
            CREATE TABLE IF NOT EXISTS global_stock_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                close_price DECIMAL(10, 2) NOT NULL,
                volume BIGINT DEFAULT 0
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_global_symbol_time 
            ON global_stock_prices(symbol, created_at DESC);
            """,
            """
            CREATE TABLE IF NOT EXISTS local_stock_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                close_price DECIMAL(10, 2) NOT NULL,
                volume BIGINT DEFAULT 0
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_local_symbol_time 
            ON local_stock_prices(symbol, created_at DESC);
            """,
            """
            CREATE TABLE IF NOT EXISTS global_predictions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_price DECIMAL(10, 2),
                forecast_json JSONB,
                explanation_text TEXT
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_global_pred_symbol_time 
            ON global_predictions(symbol, created_at DESC);
            """,
            """
            CREATE TABLE IF NOT EXISTS local_predictions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_price DECIMAL(10, 2),
                forecast_json JSONB,
                explanation_text TEXT
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_local_pred_symbol_time 
            ON local_predictions(symbol, created_at DESC);
            """
        ]

        try:
            with self.conn.cursor() as cur:
                for query in queries:
                    cur.execute(query)
            print("✅ Tables created/verified successfully")
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")

    def insert_price(self, symbol: str, price: float, volume: int, is_global: bool, created_at: datetime = None):
        """Insert a new price record."""
        if not self.conn or self.conn.closed:
            self.connect()

        if not self.conn:
            print(f"❌ Cannot insert price for {symbol} - no database connection")
            return

        table = "global_stock_prices" if is_global else "local_stock_prices"
        
        if created_at:
            query = f"""
                INSERT INTO {table} (symbol, close_price, volume, created_at)
                VALUES (%s, %s, %s, %s)
            """
            params = (symbol, price, volume, created_at)
        else:
            query = f"""
                INSERT INTO {table} (symbol, close_price, volume)
                VALUES (%s, %s, %s)
            """
            params = (symbol, price, volume)

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
        except Exception as e:
            print(f"❌ Failed to insert price for {symbol}: {e}")

    def insert_prediction(self, symbol: str, predicted_price: float, forecast_json: dict, explanation: str, is_global: bool):
        """Insert a new prediction record."""
        if not self.conn or self.conn.closed:
            self.connect()

        if not self.conn:
            print(f"❌ Cannot insert prediction for {symbol} - no database connection")
            return

        table = "global_predictions" if is_global else "local_predictions"
        query = f"""
            INSERT INTO {table} (symbol, predicted_price, forecast_json, explanation_text)
            VALUES (%s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (symbol, predicted_price, json.dumps(forecast_json), explanation))
            print(f"✅ Saved prediction for {symbol}")
        except Exception as e:
            print(f"❌ Failed to insert prediction for {symbol}: {e}")

    def get_last_update_time(self, symbol: str, is_global: bool):
        """Get the timestamp of the last entry for a symbol."""
        if not self.conn or self.conn.closed:
            self.connect()

        if not self.conn:
            return None

        table = "global_stock_prices" if is_global else "local_stock_prices"
        query = f"SELECT created_at FROM {table} WHERE symbol = %s ORDER BY created_at DESC LIMIT 1"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (symbol,))
                result = cur.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            print(f"❌ Failed to get last update time for {symbol}: {e}")
        
        return None

    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("🔌 Database connection closed")


db=DBManager()