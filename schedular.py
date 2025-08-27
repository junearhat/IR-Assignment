import schedule
import time
import logging
from datetime import datetime
from crawler import crawl
from indexer import InvertedIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

def run_pipeline():
    """Run the crawler and indexer pipeline"""
    try:
        logging.info("Starting weekly data pipeline")
        
        # Run the crawler
        logging.info("Starting crawler...")
        crawl()
        logging.info("Crawler completed successfully")
        
        # Run the indexer
        logging.info("Starting indexer...")
        indexer = InvertedIndexer()
        indexer.build_index('coventry_publications.json')
        indexer.save_index()
        logging.info("Indexer completed successfully")
        
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}", exc_info=True)

def main():
    # Schedule the job to run every Monday at 1:00 AM
    schedule.every().monday.at("01:00").do(run_pipeline)
    
    logging.info("Scheduler started - Will run every Monday at 01:00")
    
    # Run the job immediately on startup
    run_pipeline()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()