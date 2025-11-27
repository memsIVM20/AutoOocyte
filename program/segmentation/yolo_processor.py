import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

class YOLOProcessor:
    def __init__(self, model_path="best.pt"):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = YOLO(model_path)
            self.pixel_to_um = 1.25361507557
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def process_image(self, image_path, save_path, json_path):
        try:
            # Check if input image exists
            if not os.path.exists(image_path):
                self.logger.error(f"Input image not found: {image_path}")
                return False

            # Create output directories if they don't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            # Run inference
            results = self.model.predict(image_path, save=False)
            
            if len(results) == 0:
                self.logger.warning(f"No detection for image: {image_path}")
                return False

            result = results[0]
            
            # Get masks for nucleus (class 0) and cytoplasm (class 1)
            masks = result.masks
            if masks is None:
                self.logger.warning(f"No masks detected for image: {image_path}")
                return False

            # Combine masks to calculate total area
            combined_mask = np.zeros_like(masks.data[0].cpu().numpy(), dtype=np.uint8)
            for mask, cls in zip(masks.data, result.boxes.cls):
                combined_mask = cv2.bitwise_or(combined_mask, mask.cpu().numpy().astype(np.uint8))

            # Calculate area
            pixel_area = np.sum(combined_mask)
            um_area = round(pixel_area * (self.pixel_to_um ** 2))

            # Save visualization
            result.save(save_path)

            # Save area to JSON
            with open(json_path, 'w') as f:
                json.dump({"area": str(um_area)}, f, indent=4)

            return True

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return False