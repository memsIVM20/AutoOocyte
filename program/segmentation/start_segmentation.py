import logging
from watchdog.observers import Observer
from .directory_watcher import TimelapseHandler
from .yolo_processor import YOLOProcessor
import os
from pathlib import Path

def start_segmentation():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize YOLO processor
        yolo_processor = YOLOProcessor()

        # Set up the event handler and observer
        event_handler = TimelapseHandler(yolo_processor)
        observer = Observer()

        # Start watching the data directory
        data_dir = Path("data")
        observer.schedule(event_handler, str(data_dir), recursive=True)
        observer.start()

        logger.info("ML processor started successfully")
        return observer

    except Exception as e:
        logger.error(f"Failed to start ML processor: {str(e)}")
        raise