import cv2
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


def validate_file_extension(filename: str) -> bool:
    """Validate if the file has an allowed extension"""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def validate_file_size(file_content: bytes, max_size_mb: int = 10) -> bool:
    """Validate if the file size is within the allowed limit"""
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    return len(file_content) <= max_size_bytes


def preprocess_image(image_data: Union[bytes, np.ndarray]) -> np.ndarray:
    """
    Preprocess image for model prediction
    """
    try:
        if isinstance(image_data, bytes):
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 128x128 (match model's input size)
        img = cv2.resize(img, (128, 128))
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        logger.debug(f"Image shape: {img.shape}")
        logger.debug(f"Value range: {img.min():.3f} to {img.max():.3f}")

        return img

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError(f"Error preprocessing image: {e}")