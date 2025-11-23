from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from app.etl_pipeline import StockETLPipeline

def scheduled_job(pipeline):
    """
    Scheduled job that runs the ETL pipeline.
    
    Args:
        pipeline (StockETLPipeline): Initialized ETL pipeline instance
    """
    print(f"\n{'='*60}")
    print(f"⏰ Scheduler Job Started at {datetime.now()}")
    print(f"{'='*60}")
    
    # Run the complete ETL pipeline
    pipeline.run_pipeline()
    
    print(f"{'='*60}")
    print(f"✅ Job Finished at {datetime.now()}")
    print(f"{'='*60}\n")

def start_scheduler():
    """
    Start the scheduler service.
    Runs the ETL pipeline every 15 minutes.
    """
    print("🚀 Initializing Scheduler Service...")
    
    # Initialize ETL Pipeline once
    pipeline = StockETLPipeline()
    
    # Create scheduler
    scheduler = BlockingScheduler()
    
    # Schedule job every 15 minutes
    scheduler.add_job(
        lambda: scheduled_job(pipeline),
        'interval',
        minutes=15,
        id='etl_pipeline_job'
    )
    
    print("✅ Scheduler configured (Running every 15 minutes)")
    print("   Press Ctrl+C to stop.\n")
    
    # Run once immediately on startup
    print("▶️ Running initial pipeline execution...")
    scheduled_job(pipeline)
    
    # Start scheduler
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Scheduler stopped.")

if __name__ == "__main__":
    start_scheduler()
