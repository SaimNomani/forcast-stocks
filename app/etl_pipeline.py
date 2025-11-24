import os
import sys
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))

from app.db_manager import db
from app.config import settings

# Conditional imports
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available. Prediction functionality will be disabled.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("⚠ pandas_ta not found. Using manual calculation for indicators.")


class StockETLPipeline:
    """
    Formal ETL Pipeline for Stock Forecasting System.
    
    This class implements a clear Extract-Transform-Load pattern:
    - EXTRACT: Fetch stock data from YFinance
    - TRANSFORM: Calculate indicators, scale data, run predictions
    - LOAD: Save actual prices and predictions to PostgreSQL database
    """
    
    def _init_(self):
        """
        Initialize the ETL Pipeline.
        - Connect to Database
        - Define Model Paths
        - Load ML Models
        """
        print("🔧 Initializing StockETLPipeline...")
        
        # Database Connection
        db.connect()
        db.create_tables()
        
        # Define Paths (CRITICAL: Match exact folder structure)
        self.MODELS_DIR = "models"
        self.GLOBAL_SCALERS_DIR = os.path.join(self.MODELS_DIR, "global_scalers")
        self.LOCAL_SCALERS_DIR = os.path.join(self.MODELS_DIR, "local_scalers")
        self.GLOBAL_MODEL_PATH = os.path.join(self.MODELS_DIR, "global_model.h5")
        self.LOCAL_MODEL_PATH = os.path.join(self.MODELS_DIR, "local_model.h5")
        
        # Load Models (Cache them to avoid reloading)
        if TENSORFLOW_AVAILABLE:
            self.global_model = self._load_model(self.GLOBAL_MODEL_PATH, "Global")
            self.local_model = self._load_model(self.LOCAL_MODEL_PATH, "Local")
        else:
            self.global_model = None
            self.local_model = None
            print("⚠ Models not loaded - TensorFlow unavailable")
        
        print("✅ ETL Pipeline initialized successfully.\n")

    def _load_model(self, path, name):
        """Helper to load Keras models safely."""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        if os.path.exists(path):
            try:
                model = load_model(path)
                print(f"✅ {name} model loaded from {path}")
                return model
            except Exception as e:
                print(f"❌ Failed to load {name} model: {e}")
                return None
        else:
            print(f"⚠ {name} model not found at {path}")
            return None

    # ==================== STEP 1: EXTRACT ====================
    
    def extract_data(self, symbols, is_global=True):
        """
        STEP 1: EXTRACT
        
        Fetch latest stock data from YFinance for the given symbols.
        
        Args:
            symbols (list): List of stock symbols (e.g., ['AAPL', 'HUBC.KA'])
            is_global (bool): True for global stocks (15min, 50 timesteps), False for local (1day, 60 timesteps)
        
        Returns:
            dict: {symbol: DataFrame} containing historical data
        """
        # Global: 50 timesteps * 15min + buffer = ~7 days of 15min data
        # Local: 60 timesteps * 1day + 14 for indicators + buffer = ~6 months
        if is_global:
            period = "7d"
            interval = "15m"
            timesteps = 50
        else:
            period = "6mo"
            interval = "1d"
            timesteps = 60
        
        print(f"\n📥 [EXTRACT] Fetching data for {len(symbols)} symbols (Period: {period}, Interval: {interval}, Timesteps: {timesteps})...")
        data_map = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if not hist.empty:
                    data_map[symbol] = hist
                    print(f"   ✓ Fetched {len(hist)} rows for {symbol}")
                else:
                    print(f"   ⚠ No data found for {symbol}")
            except Exception as e:
                print(f"   ❌ Error fetching {symbol}: {e}")
        
        return data_map
    # ==================== STEP 2: TRANSFORM ====================
    
    def transform_data(self, raw_data, symbol, is_global):
        """
        STEP 2: TRANSFORM
        
        Process raw stock data:
        1. Calculate technical indicators (RSI_14, SMA_14)
        2. Load symbol-specific scaler from models/ folder
        3. Scale the data
        4. Prepare for prediction
        
        Args:
            raw_data (DataFrame): Raw stock data from YFinance
            symbol (str): Stock symbol
            is_global (bool): True for global stocks (50 timesteps), False for local (60 timesteps)
        
        Returns:
            tuple: (scaled_data, explanation_data, scaler) or (None, None, None) on error
        """
        print(f"🔄 [TRANSFORM] Processing {symbol}...")
        
        df = raw_data.copy()
        timesteps = 50 if is_global else 60
        
        # --- Action 1: Calculate Indicators ---
        try:
            if 'Close' not in df.columns:
                print(f"   ❌ 'Close' column missing for {symbol}")
                return None, None, None

            # Calculate SMA_14 and RSI_14
            if PANDAS_TA_AVAILABLE:
                df['SMA_14'] = ta.sma(df['Close'], length=14)
                df['RSI_14'] = ta.rsi(df['Close'], length=14)
            else:
                # Manual calculation fallback
                df['SMA_14'] = df['Close'].rolling(window=14).mean()
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Drop NaNs created by indicators
            df.dropna(inplace=True)
            
            if len(df) < timesteps:
                print(f"   ⚠ Insufficient data after indicators for {symbol} (Rows: {len(df)}, Need: {timesteps})")
                return None, None, None

        except Exception as e:
            print(f"   ❌ Error calculating indicators for {symbol}: {e}")
            return None, None, None

        # --- Action 2: Load Scaler (CRITICAL) ---
        safe_symbol = symbol.replace(".", "_")
        scaler_filename = f"{safe_symbol}_scaler.pkl"
        directory = self.GLOBAL_SCALERS_DIR if is_global else self.LOCAL_SCALERS_DIR
        scaler_path = os.path.join(directory, scaler_filename)
        
        if not os.path.exists(scaler_path):
            print(f"   ⚠ Scaler not found: {scaler_path}. Skipping {symbol}.")
            return None, None, None
        
        try:
            scaler = joblib.load(scaler_path)
            
            # Prepare features for scaling
            df['Log_Volume'] = np.log1p(df['Volume'])
            
            # Determine features based on scaler
            if hasattr(scaler, 'n_features_in_'):
                if scaler.n_features_in_ == 4:
                    features_to_scale = df[['Close', 'Log_Volume', 'RSI_14', 'SMA_14']].values
                elif scaler.n_features_in_ == 1:
                    features_to_scale = df[['Close']].values
                else:
                    print(f"   ⚠ Scaler expects {scaler.n_features_in_} features, unsupported.")
                    return None, None, None
            else:
                features_to_scale = df[['Close']].values
            
            scaled_data = scaler.transform(features_to_scale)
            
            # Get last N timesteps for prediction (50 for global, 60 for local)
            if len(scaled_data) < timesteps:
                print(f"   ⚠ Insufficient scaled data for {symbol} (Have: {len(scaled_data)}, Need: {timesteps})")
                return None, None, None
                
            last_n_scaled = scaled_data[-timesteps:]
            
            # Prepare explanation data
            latest_row = df.iloc[-1]
            explanation_data = {
                'Close': float(latest_row['Close']),
                'SMA_14': float(latest_row['SMA_14']),
                'RSI_14': float(latest_row['RSI_14']),
                'Volume': int(latest_row['Volume'])
            }
            
            return last_n_scaled, explanation_data, scaler

        except Exception as e:
            print(f"   ❌ Error scaling data for {symbol}: {e}")
            return None, None, None


    def _predict(self, model, scaled_data, scaler):
        """
        Internal prediction logic.
        Predicts next 10 days using the LSTM model.
        
        Args:
            model: Loaded Keras model
            scaled_data: Scaled input sequence (50 or 60 timesteps)
            scaler: Scaler object for inverse transformation
        
        Returns:
            dict: {predicted_price, forecast_json} or None on error
        """
        if not TENSORFLOW_AVAILABLE or model is None:
            print("   ⚠ Cannot predict - model not available")
            return None
            
        try:
            n_features = scaled_data.shape[1]
            timesteps = scaled_data.shape[0]  # Dynamic: 50 or 60
            input_seq = scaled_data.reshape(1, timesteps, n_features)
            
            predictions = []
            current_batch = input_seq.copy()
            
            # Predict next 10 days iteratively
            for _ in range(10):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                
                # Update batch for next prediction
                pred_reshaped = np.array(pred).reshape(1, 1, -1)
                
                if pred_reshaped.shape[2] != n_features:
                    new_step = current_batch[:, -1:, :].copy()
                    new_step[0, 0, 0] = pred[0]
                    current_batch = np.append(current_batch[:, 1:, :], new_step, axis=1)
                else:
                    current_batch = np.append(current_batch[:, 1:, :], pred_reshaped, axis=1)

            # Inverse transform predictions
            predictions = np.array(predictions)
            
            if scaler.n_features_in_ > 1:
                dummy = np.zeros((10, scaler.n_features_in_))
                dummy[:, 0] = predictions[:, 0]
                inversed = scaler.inverse_transform(dummy)[:, 0]
            else:
                inversed = scaler.inverse_transform(predictions).flatten()
            
            forecast_json = {
                f"day_{i+1}": round(float(val), 2)
                for i, val in enumerate(inversed)
            }
            
            return {
                "predicted_price": float(inversed[0]),
                "forecast_json": forecast_json
            }

        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            return None


    def _generate_explanation(self, data):
        """
        Generate rule-based explanation based on technical indicators.
        
        Args:
            data (dict): Contains Close, SMA_14, RSI_14
        
        Returns:
            str: Human-readable explanation
        """
        explanation = []
        price = data['Close']
        sma = data['SMA_14']
        rsi = data['RSI_14']
        
        if price > sma:
            explanation.append("Price is above SMA 14 (Bullish).")
        else:
            explanation.append("Price is below SMA 14 (Bearish).")
        
        if rsi > 70:
            explanation.append("RSI is overbought (>70).")
        elif rsi < 30:
            explanation.append("RSI is oversold (<30).")
        else:
            explanation.append(f"RSI is neutral ({round(rsi, 2)}).")
        
        return " ".join(explanation)

    # ==================== STEP 3: LOAD ====================
    
    def load_predictions(self, symbol, current_data, prediction_data, is_global):
        """
        STEP 3: LOAD
        
        Save results to PostgreSQL database:
        1. Insert actual price data into *_stock_prices table
        2. Insert prediction data into *_predictions table
        
        Args:
            symbol (str): Stock symbol
            current_data (dict): Current actual data (Close, Volume)
            prediction_data (dict): Prediction results
            is_global (bool): True for global stocks, False for local
        """
        print(f"💾 [LOAD] Saving results for {symbol}...")
        
        try:
            # 1. Insert Actual Data
            close_price = float(current_data['Close'])
            volume = int(current_data.get('Volume', 0))
            
            db.insert_price(symbol, close_price, volume, is_global)
            
            # 2. Insert Predicted Data
            db.insert_prediction(
                symbol,
                prediction_data['predicted_price'],
                prediction_data['forecast_json'],
                prediction_data['explanation_text'],
                is_global
            )
            
            print(f"   ✓ Saved to database successfully")
            
        except Exception as e:
            print(f"   ❌ Error loading data to DB for {symbol}: {e}")

    # ==================== ORCHESTRATOR ====================
    def run_pipeline(self):
        """
        Main orchestrator that runs the complete ETL pipeline.
        
        Logic:
        - Global stocks: Always run (every 15 minutes)
        - Local stocks: Only run if last update was >24 hours ago
        """
        print(f"\n⏰ Pipeline started at {datetime.now()}")
        print("=" * 60)
        
        # Process Global Stocks
        print("\n🌍 Processing Global Stocks (Always Run)...")
        global_data_map = self.extract_data(settings.global_stocks, is_global=True)
        
        for symbol, df in global_data_map.items():
            self._process_symbol(symbol, df, is_global=True)
        
        # Process Local Stocks (with 24h check)
        print("\n🇵🇰 Processing Local Stocks (24h Check)...")
        local_data_map = self.extract_data(settings.local_stocks, is_global=False)
        
        for symbol, df in local_data_map.items():
            last_update = db.get_last_update_time(symbol, is_global=False)
            
            should_update = False
            if not last_update:
                should_update = True
                print(f"   🆕 First time update for {symbol}")
            else:
                time_diff = datetime.now() - last_update
                if time_diff > timedelta(hours=settings.local_update_interval_hours):
                    should_update = True
                    print(f"   🕒 >24h since last update for {symbol} (Last: {last_update})")
                else:
                    print(f"   ⏭ Skipping {symbol} (Updated {time_diff} ago)")
            
            if should_update:
                self._process_symbol(symbol, df, is_global=False)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline run completed.\n")
    def _process_symbol(self, symbol, raw_data, is_global):
        """
        Process a single symbol through the ETL pipeline.
        
        Args:
            symbol (str): Stock symbol
            raw_data (DataFrame): Raw data from extract step
            is_global (bool): True for global, False for local
        """
        # Step 2: Transform
        scaled_data, explanation_data, scaler = self.transform_data(raw_data, symbol, is_global)
        
        if scaled_data is None:
            return
        
        # Predict
        model = self.global_model if is_global else self.local_model
        if not model:
            print(f"   ❌ No model available for {symbol}")
            return
        
        prediction_result = self._predict(model, scaled_data, scaler)
        
        if prediction_result:
            prediction_result['explanation_text'] = self._generate_explanation(explanation_data)
            self.load_predictions(symbol, explanation_data, prediction_result,is_global)