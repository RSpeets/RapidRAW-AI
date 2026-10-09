from pydantic import BaseModel, Field
from typing import Optional


class InpaintRequest(BaseModel):
    """Request format for inpainting operations.
    
    This format is compatible with RapidRAW's ai_connector.rs.
    Supports both middleware-style (source_id) and cloud-style (image_base64).
    """
    source_id: str = Field(default="", description="Hash identifier for the source image (middleware-style)")
    prompt: str = Field(..., description="Description of what to paint")
    negative_prompt: str = Field(
        default="",
        description="Description of what to avoid"
    )
    mask_image_base64: str = Field(..., description="Base64-encoded mask image")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded source image (cloud-style)")
    seed: int = Field(default=0, description="Random seed for reproducibility")


class MiddlewareResponse(BaseModel):
    """Response format for middleware-style inpainting.
    
    Compatible with RapidRAW's ai_connector.rs MiddlewareResponse.
    """
    x: int = Field(..., description="X coordinate of result")
    y: int = Field(..., description="Y coordinate of result")
    color: str = Field(..., description="Base64-encoded result image")


class CloudInpaintRequest(BaseModel):
    """Request format for cloud-style inpainting.
    
    Compatible with RapidRAW's ai_connector.rs CloudInpaintRequest.
    """
    image_base64: str = Field(..., description="Base64-encoded source image")
    mask_image_base64: str = Field(..., description="Base64-encoded mask image")
    prompt: str = Field(..., description="Description of what to paint")
    seed: int = Field(default=0, description="Random seed for reproducibility")


class CloudInpaintResponse(BaseModel):
    """Response format for cloud-style inpainting.
    
    Compatible with RapidRAW's ai_connector.rs CloudInpaintResponse.
    """
    color: str = Field(..., description="Base64-encoded result image")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(..., description="Service status")
    version: str = Field(default="0.1.0", description="Service version")


class ModelsStatusResponse(BaseModel):
    """Response for model status check."""
    available_models: dict[str, bool] = Field(
        default_factory=dict,
        description="Dictionary of model names and their availability"
    )
    device: str = Field(..., description="Current device (cuda or cpu)")
