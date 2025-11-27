import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .yolo_processor import YOLOProcessor
import logging
from pathlib import Path

class TimelapseHandler(FileSystemEventHandler):
    def __init__(self, yolo_processor):
        self.yolo_processor = yolo_processor
        self.logger = logging.getLogger(__name__)
        self.processed_files = set()

    def on_created(self, event):
        if event.is_directory:
            return
        
        if not event.src_path.endswith('.jpg'):
            return

        try:
            # Convert Windows path to proper format
            src_path = Path(event.src_path)
            
            # Check if file matches our expected pattern
            if not self._is_valid_timelapse_file(src_path):
                return

            # Generate output paths
            analyze_path = self._get_analyze_path(src_path)
            json_path = self._get_json_path(src_path)

            # Process image if not already processed
            if str(src_path) not in self.processed_files:
                success = self.yolo_processor.process_image(
                    str(src_path),
                    str(analyze_path),
                    str(json_path)
                )
                if success:
                    self.processed_files.add(str(src_path))
                    self.logger.info(f"Successfully processed: {src_path}")
                else:
                    self.logger.error(f"Failed to process: {src_path}")

        except Exception as e:
            self.logger.error(f"Error handling file {event.src_path}: {str(e)}")

    def _is_valid_timelapse_file(self, path):
        try:
            parts = path.parts
            if "data" not in parts:
                return False
            
            folder_name = parts[parts.index("data") + 1]
            if not folder_name.startswith("finalOocyte_"):
                return False

            return True
        except:
            return False

    def _get_analyze_path(self, src_path):
        parts = list(src_path.parts)
        final_idx = parts.index("finalOocyte_1" if "finalOocyte_1" in parts else "finalOocyte_2")
        parts[final_idx] = parts[final_idx].replace("finalOocyte_", "analyzeOocyte_")
        return Path(*parts)

    def _get_json_path(self, src_path):
        parts = list(src_path.parts)
        final_idx = parts.index("finalOocyte_1" if "finalOocyte_1" in parts else "finalOocyte_2")
        parts[final_idx] = parts[final_idx].replace("finalOocyte_", "json_")
        parts[-1] = parts[-1].replace(".jpg", ".json")
        return Path(*parts)