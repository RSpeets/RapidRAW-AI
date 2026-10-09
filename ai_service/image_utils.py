"""Image processing utilities compatible with RapidRAW."""

import base64
from io import BytesIO

from PIL import Image


def base64_to_image(data: str) -> Image.Image:
    """Decode base64 string to PIL Image.
    
    Args:
        data: Base64-encoded image data, optionally with data URL prefix
        
    Returns:
        PIL Image object
    """
    if data.startswith("data:"):
        # Remove data URL prefix if present
        data = data.split(",", 1)[1]
    
    image_data = base64.standard_b64decode(data)
    return Image.open(BytesIO(image_data))


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64-encoded image data
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    image_data = buffer.getvalue()
    return base64.standard_b64encode(image_data).decode("utf-8")


def image_to_base64_jpeg(image: Image.Image, quality: int = 95) -> str:
    """Encode PIL Image to base64 JPEG string.
    
    Args:
        image: PIL Image object
        quality: JPEG quality (0-100)
        
    Returns:
        Base64-encoded JPEG image data
    """
    buffer = BytesIO()
    # Convert RGBA to RGB if needed
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=quality)
    image_data = buffer.getvalue()
    return base64.standard_b64encode(image_data).decode("utf-8")
