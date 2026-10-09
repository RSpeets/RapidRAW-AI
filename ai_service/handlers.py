"""AI operation handlers with PyTorch implementations.

These handlers process AI requests and coordinate with the model manager.
"""

import logging
from typing import Optional

from ai_service.image_utils import base64_to_image, image_to_base64
from ai_service.models_manager import get_model_manager, ModelType

logger = logging.getLogger(__name__)


class InpaintingHandler:
    """Handle inpainting operations."""
    
    @staticmethod
    async def process_middleware_style(
        source_id: str,
        prompt: str,
        negative_prompt: str,
        mask_image_base64: str,
        image_base64: Optional[str] = None,
        seed: int = 0,
    ) -> tuple[int, int, str]:
        """Process inpainting - supports both middleware and cloud styles.
        
        If image_base64 is provided, use it (cloud-style).
        Otherwise, try to fetch from cache using source_id (middleware-style).
        
        Args:
            source_id: Hash of the source image (middleware-style)
            prompt: Inpainting prompt
            negative_prompt: Negative prompt
            mask_image_base64: Base64-encoded mask
            image_base64: Base64-encoded image (cloud-style, optional)
            seed: Random seed
            
        Returns:
            Tuple of (x, y, result_base64)
        """
        logger.info(f"Processing inpainting: {prompt}")
        
        try:
            manager = get_model_manager()
            
            # Determine which image to use
            source_image_b64 = image_base64
            
            if not source_image_b64:
                # Try middleware-style: fetch from cache
                source_image_b64 = manager.get_cached_image(source_id)
            
            if not source_image_b64:
                # Image not found - return special marker for RapidRAW to upload it
                logger.info(f"Image not in cache for {source_id}, returning 404 for RapidRAW to upload")
                raise ValueError("IMAGE_NOT_CACHED_404")  # Special marker
            
            result_base64 = await manager.inpaint(
                image_base64=source_image_b64,
                mask_base64=mask_image_base64,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
            )
            
            return (0, 0, result_base64)
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            raise
    
    @staticmethod
    async def process_cloud_style(
        image_base64: str,
        mask_base64: str,
        prompt: str,
        seed: int = 0,
    ) -> str:
        """Process cloud-style inpainting with full image data.
        
        Args:
            image_base64: Base64-encoded source image
            mask_base64: Base64-encoded mask
            prompt: Inpainting prompt
            seed: Random seed
            
        Returns:
            Base64-encoded result image
        """
        logger.info(f"Processing cloud-style inpainting: {prompt}")
        
        try:
            manager = get_model_manager()
            
            result_base64 = await manager.inpaint(
                image_base64=image_base64,
                mask_base64=mask_base64,
                prompt=prompt,
                negative_prompt="",
                seed=seed,
            )
            
            return result_base64
        except Exception as e:
            logger.error(f"Cloud-style inpainting failed: {e}")
            raise


class MaskGenerationHandler:
    """Handle mask generation operations."""
    
    @staticmethod
    async def generate_foreground_mask(image_base64: str) -> str:
        """Generate foreground/subject segmentation mask.
        
        Args:
            image_base64: Base64-encoded input image
            
        Returns:
            Base64-encoded foreground mask
        """
        logger.info("Generating foreground mask")
        
        try:
            manager = get_model_manager()
            result = await manager.generate_foreground_mask(image_base64)
            return result
        except Exception as e:
            logger.error(f"Foreground mask generation failed: {e}")
            raise
    
    @staticmethod
    async def generate_sky_mask(image_base64: str) -> str:
        """Generate sky segmentation mask.
        
        Args:
            image_base64: Base64-encoded input image
            
        Returns:
            Base64-encoded sky mask
        """
        logger.info("Generating sky mask")
        
        try:
            manager = get_model_manager()
            result = await manager.generate_sky_mask(image_base64)
            return result
        except Exception as e:
            logger.error(f"Sky mask generation failed: {e}")
            raise
    
    @staticmethod
    async def generate_depth_mask(image_base64: str) -> str:
        """Generate depth map from image.
        
        Args:
            image_base64: Base64-encoded input image
            
        Returns:
            Base64-encoded depth map
        """
        logger.info("Generating depth mask")
        
        try:
            manager = get_model_manager()
            result = await manager.generate_depth_mask(image_base64)
            return result
        except Exception as e:
            logger.error(f"Depth mask generation failed: {e}")
            raise
    
    @staticmethod
    async def generate_subject_mask(image_base64: str) -> str:
        """Generate subject mask using SAM (Segment Anything Model).
        
        Args:
            image_base64: Base64-encoded input image
            
        Returns:
            Base64-encoded subject mask
        """
        logger.info("Generating subject mask with SAM")
        
        try:
            manager = get_model_manager()
            
            if not manager.is_model_loaded(ModelType.SAM):
                manager.load_model(ModelType.SAM)
            
            # Placeholder: SAM implementation
            return image_base64
        except Exception as e:
            logger.error(f"Subject mask generation failed: {e}")
            raise


class EmbeddingHandler:
    """Handle image embedding generation."""
    
    @staticmethod
    async def generate_embeddings(image_base64: str) -> list[float]:
        """Generate image embeddings for similarity search.
        
        Uses CLIP model to generate image embeddings.
        
        Args:
            image_base64: Base64-encoded input image
            
        Returns:
            List of embedding values
        """
        logger.info("Generating image embeddings")
        
        try:
            manager = get_model_manager()
            result = await manager.generate_embeddings(image_base64)
            return result
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise


class HealthHandler:
    """Handle health and status checks."""
    
    @staticmethod
    def get_service_status() -> dict:
        """Get overall service status.
        
        Returns:
            Dictionary with service status information
        """
        manager = get_model_manager()
        return {
            "status": "healthy",
            "device": manager.device,
            "models_loaded": len(manager.get_loaded_models()),
        }
    
    @staticmethod
    def get_models_status() -> dict[str, bool]:
        """Get status of all available models.
        
        Returns:
            Dictionary mapping model names to loaded status
        """
        manager = get_model_manager()
        
        # Return status of all known model types
        all_models = {model_type.value: False for model_type in ModelType}
        all_models.update(manager.get_loaded_models())
        
        return all_models
