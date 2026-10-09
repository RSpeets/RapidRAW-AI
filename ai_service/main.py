"""Main FastAPI application for the AI Service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse

from ai_service.config import settings
from ai_service.handlers import (
    HealthHandler,
    InpaintingHandler,
    MaskGenerationHandler,
    EmbeddingHandler,
)
from ai_service.models import (
    CloudInpaintRequest,
    CloudInpaintResponse,
    HealthResponse,
    InpaintRequest,
    MiddlewareResponse,
    ModelsStatusResponse,
)
from ai_service.models_manager import init_model_manager, get_model_manager

logger = logging.getLogger(__name__)


# Global state for models (will be initialized on startup)
class AppState:
    """Application state for managing resources."""
    models_ready: bool = False
    error_message: str = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("AI Service starting up...")
    try:
        # Initialize model manager
        init_model_manager(device=settings.device)
        app_state.models_ready = True
        logger.info("AI Service ready")
    except Exception as e:
        app_state.error_message = str(e)
        logger.error(f"Startup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("AI Service shutting down...")


app = FastAPI(
    title="RapidRAW AI Service",
    description="AI backend for RapidRAW image processing",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure logging
logging.basicConfig(level=logging.INFO)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Compatible with RapidRAW's ai_connector.rs check_status() function.
    """
    if not app_state.models_ready:
        raise HTTPException(
            status_code=503,
            detail=app_state.error_message or "Service not ready"
        )
    
    return HealthResponse(status="healthy")


@app.post("/cache/image")
async def cache_image(request: dict) -> dict:
    """Cache an image for later use in inpainting.
    
    Stores the source image indexed by source_id for middleware-style inpainting.
    
    Args:
        request: Dict with "source_id" and "image_base64" fields
        
    Returns:
        Status confirmation
    """
    try:
        source_id = request.get("source_id")
        image_base64 = request.get("image_base64")
        
        if not source_id or not image_base64:
            raise ValueError("Missing source_id or image_base64")
        
        manager = get_model_manager()
        manager.cache_image(source_id, image_base64)
        
        logger.info(f"Cached image with source_id: {source_id}")
        return {"status": "cached", "source_id": source_id}
    except Exception as e:
        logger.error(f"Cache image failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload_source")
async def upload_source(
    source_id: str = Form(...),
    file: UploadFile = File(None),
) -> dict:
    """Upload source image for inpainting (multipart form data).
    
    This endpoint is called by RapidRAW's upload_source_image() function.
    Accepts multipart form data with:
    - source_id: Hash identifier for the image
    - file: The actual JPEG image file
    
    Stores the image indexed by source_id for middleware-style inpainting.
    
    Returns:
        Status confirmation
    """
    try:
        import base64
        
        if not source_id:
            raise ValueError("Missing source_id")
        
        if not file:
            raise ValueError("Missing file upload")
        
        logger.info(f"Uploading source image: {file.filename} for source_id={source_id}")
        
        # Read file bytes and encode to base64
        image_bytes = await file.read()
        img_data = base64.b64encode(image_bytes).decode('utf-8')
        
        manager = get_model_manager()
        manager.cache_image(source_id, img_data)
        
        logger.info(f"✓ Uploaded and cached source image: {source_id} ({len(image_bytes)} bytes)")
        return {"status": "uploaded", "source_id": source_id}
    except Exception as e:
        logger.error(f"Upload source failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/status", response_model=ModelsStatusResponse)
async def models_status() -> ModelsStatusResponse:
    """Get status of available models.
    
    Returns information about loaded and available models.
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    handler = HealthHandler()
    models = handler.get_models_status()
    
    return ModelsStatusResponse(
        available_models=models,
        device=settings.device,
    )


@app.post("/inpaint", response_model=MiddlewareResponse)
async def inpaint(request: InpaintRequest) -> MiddlewareResponse:
    """Perform inpainting on an image.
    
    Compatible with RapidRAW's ai_connector.rs process_inpainting() function.
    Supports both:
    - Middleware-style: source_id + mask (requires prior /cache/image call)
    - Cloud-style: image_base64 + mask (send image directly in request)
    
    If middleware-style image not found, returns 404 to signal RapidRAW to upload it.
    
    Args:
        request: InpaintRequest with (source_id or image_base64) + mask + prompt
        
    Returns:
        MiddlewareResponse with (x, y, color) fields
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Inpainting request: {request.prompt}")
        
        x, y, result_b64 = await InpaintingHandler.process_middleware_style(
            source_id=request.source_id,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            mask_image_base64=request.mask_image_base64,
            image_base64=request.image_base64,  # Accept cloud-style image too
            seed=request.seed,
        )
        
        return MiddlewareResponse(x=x, y=y, color=result_b64)
    except ValueError as e:
        # Special handling for IMAGE_NOT_CACHED_404 - signal RapidRAW to upload
        if "IMAGE_NOT_CACHED_404" in str(e):
            raise HTTPException(status_code=404, detail="Source image not cached. Please upload via /cache/image")
        logger.error(f"Inpainting failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inpainting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inpaint/cloud", response_model=CloudInpaintResponse)
async def inpaint_cloud(request: CloudInpaintRequest) -> CloudInpaintResponse:
    """Perform cloud-style inpainting on an image.
    
    Compatible with RapidRAW's ai_connector.rs process_cloud_inpainting() function.
    Accepts the cloud-style request format with full image data.
    
    Args:
        request: CloudInpaintRequest with image, mask, and prompt
        
    Returns:
        CloudInpaintResponse with base64-encoded result
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info(f"Cloud inpainting request: {request.prompt}")
        
        result_b64 = await InpaintingHandler.process_cloud_style(
            image_base64=request.image_base64,
            mask_base64=request.mask_image_base64,
            prompt=request.prompt,
            seed=request.seed,
        )
        
        return CloudInpaintResponse(color=result_b64)
    except Exception as e:
        logger.error(f"Cloud inpainting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mask generation endpoints
@app.post("/masks/foreground")
async def generate_foreground_mask(request: dict):
    """Generate foreground/subject segmentation mask.
    
    Args:
        request: Dict with "image_base64" field
        
    Returns:
        Dict with "mask_base64" field
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        image_b64 = request.get("image_base64")
        if not image_b64:
            raise ValueError("image_base64 required")
        
        mask_b64 = await MaskGenerationHandler.generate_foreground_mask(image_b64)
        return {"mask_base64": mask_b64}
    except Exception as e:
        logger.error(f"Foreground mask generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/masks/sky")
async def generate_sky_mask(request: dict):
    """Generate sky segmentation mask.
    
    Args:
        request: Dict with "image_base64" field
        
    Returns:
        Dict with "mask_base64" field
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        image_b64 = request.get("image_base64")
        if not image_b64:
            raise ValueError("image_base64 required")
        
        mask_b64 = await MaskGenerationHandler.generate_sky_mask(image_b64)
        return {"mask_base64": mask_b64}
    except Exception as e:
        logger.error(f"Sky mask generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/masks/depth")
async def generate_depth_mask(request: dict):
    """Generate depth map from image.
    
    Args:
        request: Dict with "image_base64" field
        
    Returns:
        Dict with "mask_base64" field (depth map)
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        image_b64 = request.get("image_base64")
        if not image_b64:
            raise ValueError("image_base64 required")
        
        mask_b64 = await MaskGenerationHandler.generate_depth_mask(image_b64)
        return {"mask_base64": mask_b64}
    except Exception as e:
        logger.error(f"Depth mask generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/masks/subject")
async def generate_subject_mask(request: dict):
    """Generate subject mask using SAM (Segment Anything Model).
    
    Args:
        request: Dict with "image_base64" field
        
    Returns:
        Dict with "mask_base64" field
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        image_b64 = request.get("image_base64")
        if not image_b64:
            raise ValueError("image_base64 required")
        
        mask_b64 = await MaskGenerationHandler.generate_subject_mask(image_b64)
        return {"mask_base64": mask_b64}
    except Exception as e:
        logger.error(f"Subject mask generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings")
async def generate_embeddings(request: dict):
    """Generate image embeddings for similarity search.
    
    Args:
        request: Dict with "image_base64" field
        
    Returns:
        Dict with "embeddings" field (list of floats)
    """
    if not app_state.models_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        image_b64 = request.get("image_base64")
        if not image_b64:
            raise ValueError("image_base64 required")
        
        embeddings = await EmbeddingHandler.generate_embeddings(image_b64)
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
