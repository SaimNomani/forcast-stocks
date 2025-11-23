import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.etl_pipeline import StockETLPipeline
from app.config import settings

def test_pipeline():
    print("🧪 Starting ETL Pipeline Test...")
    print("="*60)
    
    # Initialize the ETL Pipeline
    print("\n1️⃣ Initializing ETL Pipeline...")
    pipeline = StockETLPipeline()
    
    # Test with a small subset of symbols
    print("\n2️⃣ Testing with configured symbols...")
    print(f"   Global stocks: {settings.global_stocks}")
    print(f"   Local stocks: {settings.local_stocks}")
    
    # Run the pipeline
    print("\n3️⃣ Running ETL Pipeline...")
    try:
        pipeline.run_pipeline()
        print("\n✅ Pipeline test completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)

if __name__ == "__main__":
    test_pipeline()
