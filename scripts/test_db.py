import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db_manager import db
from app.config import settings

def test_db():
    print("🧪 Testing Database Connection...")
    print(f"URL: {settings.database_url.split('@')[-1]}") # Print host/db only for security

    try:
        db.connect()
        if db.conn:
            print("✅ Connection Successful!")
            db.create_tables()
            print("✅ Tables Verified.")
        else:
            print("❌ Connection Failed.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_db()
