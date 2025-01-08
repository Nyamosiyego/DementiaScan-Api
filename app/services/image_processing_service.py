import io
import logging

import cv2
from PIL import Image
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


class ImageProcessingService:
    @staticmethod
    def process_image(image_bytes: bytes) -> np.ndarray:
        """Process image bytes into a numpy array using Pillow"""
        try:
            # Validate image
            img = Image.open(io.BytesIO(image_bytes))
            logger.debug(f"Original image size: {img.size}")  # Log original size
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize((224, 224))  # Resize to model input size

            # Convert to numpy array, normalize and convert to float32
            img_array = np.asarray(img) / 255.0
            img_array = img_array.astype(np.float32)  # Convert to float32
            logger.debug(f"Processed image shape: {img_array.shape}, dtype: {img_array.dtype}")

            return img_array
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise ValueError("Image processing failed")

    @staticmethod
    def validate_image(image_array: np.ndarray) -> bool:
        """Validate the processed image array"""
        try:
            # Check shape
            if len(image_array.shape) != 4:
                logger.error(f"Invalid shape: expected 4 dimensions, got {len(image_array.shape)}")
                return False

            # Check dimensions
            if image_array.shape[1:3] != (128, 128):
                logger.error(f"Invalid dimensions: expected (128, 128), got {image_array.shape[1:3]}")
                return False

            # Check channels
            if image_array.shape[3] != 3:
                logger.error(f"Invalid number of channels: expected 3, got {image_array.shape[3]}")
                return False

            # Check dtype
            if image_array.dtype != np.float32:
                logger.error(f"Invalid dtype: expected float32, got {image_array.dtype}")
                return False

            # Check value range
            if image_array.min() < 0 or image_array.max() > 1:
                logger.error(f"Invalid value range: values should be between 0 and 1")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating image array: {str(e)}")
            return False