# run.py
import uvicorn
from pathlib import Path
import logging
from config.settings import LOG_DIR, LOG_FILE, LOG_FORMAT, LOG_LEVEL


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function to run the application"""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting application...")

    # Check if model exists
    from config.settings import MODEL_PATH
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        logger.error("Please place your trained model file at the specified location")
        return

    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()