import psycopg2
from psycopg2.extras import RealDictCursor
from app.config import settings
import json
from datetime import datetime

class DBManager:
    def __init__(self):
        self.conn = None

    def connect(self):
        """Establish a connection to the database."""
        try:
            self.conn = psycopg2.connect(settings.database_url)
            self.conn.autocommit = True
            print("✅ Connected to Database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.conn = None

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        if not self.conn:
            self.connect()
        
        if not self.conn:
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
            CREATE TABLE IF NOT EXISTS local_stock_prices (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                close_price DECIMAL(10, 2) NOT NULL,
                volume BIGINT DEFAULT 0
            );
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
            CREATE TABLE IF NOT EXISTS local_predictions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_price DECIMAL(10, 2),
                forecast_json JSONB,
                explanation_text TEXT
            );
            """
        ]

        try:
            with self.conn.cursor() as cur:
                for query in queries:
                    cur.execute(query)
            print("✅ Tables created/verified.")
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")

    def insert_price(self, symbol: str, price: float, volume: int, is_global: bool):
        """Insert a new price record."""
        if not self.conn:
            self.connect()

        table = "global_stock_prices" if is_global else "local_stock_prices"
        query = f"""
            INSERT INTO {table} (symbol, close_price, volume)
            VALUES (%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (symbol, price, volume))
            print(f"✅ Saved price for {symbol} in {table}")
        except Exception as e:
            print(f"❌ Failed to insert price for {symbol}: {e}")

    def insert_prediction(self, symbol: str, predicted_price: float, forecast_json: dict, explanation: str, is_global: bool):
        """Insert a new prediction record."""
        if not self.conn:
            self.connect()

        table = "global_predictions" if is_global else "local_predictions"
        query = f"""
            INSERT INTO {table} (symbol, predicted_price, forecast_json, explanation_text)
            VALUES (%s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (symbol, predicted_price, json.dumps(forecast_json), explanation))
            print(f"✅ Saved prediction for {symbol} in {table}")
        except Exception as e:
            print(f"❌ Failed to insert prediction for {symbol}: {e}")

    def get_last_update_time(self, symbol: str, is_global: bool):
        """Get the timestamp of the last entry for a symbol."""
        if not self.conn:
            self.connect()

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

db = DBManager()
