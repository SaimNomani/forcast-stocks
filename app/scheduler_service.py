import sys
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))

from app.etl_pipeline import StockETLPipeline
from app.config import settings


def scheduled_job(pipeline):
    """
    Scheduled job that runs the ETL pipeline.
    
    Args:
        pipeline (StockETLPipeline): Initialized ETL pipeline instance
    """
    print(f"\n{'='*60}")
    print(f"⏰ Scheduler Job Started at {datetime.now()}")
    print(f"{'='*60}")
    
    try:
        # Run the complete ETL pipeline
        pipeline.run_pipeline()
    except Exception as e:
        print(f"❌ Error in scheduled job: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}")
    print(f"✅ Job Finished at {datetime.now()}")
    print(f"{'='*60}\n")


def start_scheduler():
    """
    Start the scheduler service.
    Runs the ETL pipeline at configured intervals.
    """
    print("🚀 Initializing Scheduler Service...")
    
    try:
        # Initialize ETL Pipeline once
        pipeline = StockETLPipeline()
        
        # Create scheduler
        scheduler = BlockingScheduler()
        
        # Schedule job based on config
        interval_minutes = settings.fetch_interval_seconds // 60
        scheduler.add_job(
            lambda: scheduled_job(pipeline),
            'interval',
            minutes=interval_minutes,
            id='etl_pipeline_job',
            max_instances=1  # Prevent overlapping runs
        )
        
        print(f"✅ Scheduler configured (Running every {interval_minutes} minutes)")
        print("   Press Ctrl+C to stop.\n")
        
        # Run once immediately on startup
        print("▶ Running initial pipeline execution...")
        scheduled_job(pipeline)
        
        # Start scheduler
        print("\n⏰ Starting scheduler loop...")
        scheduler.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped by user.")
    except Exception as e:
        print(f"\n❌ Scheduler error: {e}")
        import traceback
        traceback.print_exc()


if _name_ == "_main_":
    start_scheduler()