"""Model management for AI operations.

This module handles loading, caching, and managing ML models with PyTorch.
"""

from enum import Enum
from typing import Optional, Any
import logging
import io
from PIL import Image

try:
    import torch
    import torchvision.transforms as transforms
    from diffusers import StableDiffusionInpaintPipeline
    from transformers import AutoModelForImageSegmentation, AutoProcessor
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from ai_service.image_utils import base64_to_image, image_to_base64
from ai_service.config import settings

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available model types."""
    INPAINT = "inpaint"
    DEPTH = "depth"
    SAM = "sam"
    U2NET = "u2net"
    SKY_SEGMENTATION = "sky_seg"


class ModelManager:
    """Manages ML model lifecycle and inference with PyTorch.
    
    Handles loading, caching, and running inference for various ML models.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not installed. Using placeholder implementations only.")
        
        self.models: dict[ModelType, Any] = {}
        self.device = "cpu"
        self.logger = logging.getLogger(__name__)
        self.image_cache: dict[str, str] = {}  # Cache source images by source_id
        self._detect_device()
    
    def _detect_device(self):
        """Auto-detect available device."""
        if not PYTORCH_AVAILABLE:
            self.device = "cpu"
            self.logger.info("Device: cpu (PyTorch not available)")
            return
        
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                self.logger.info("CUDA not available, using CPU")
        except Exception as e:
            self.device = "cpu"
            self.logger.warning(f"Error detecting device: {e}, using CPU")
    
    def set_device(self, device: str):
        """Set the compute device (cuda or cpu).
        
        Args:
            device: Device name ("cuda" or "cpu")
        """
        if device == "auto":
            self._detect_device()
        else:
            self.device = device
        self.logger.info(f"Model device set to: {self.device}")
    
    def cache_image(self, source_id: str, image_base64: str) -> None:
        """Cache an image for later inpainting.
        
        Args:
            source_id: Unique identifier for the image
            image_base64: Base64-encoded image
        """
        self.image_cache[source_id] = image_base64
        self.logger.debug(f"Cached image: {source_id} ({len(image_base64)} bytes)")
    
    def get_cached_image(self, source_id: str) -> Optional[str]:
        """Retrieve a cached image.
        
        Args:
            source_id: Unique identifier for the image
            
        Returns:
            Base64-encoded image or None if not found
        """
        if source_id not in self.image_cache:
            self.logger.warning(f"Image not found in cache: {source_id}")
            return None
        return self.image_cache[source_id]
    
    def load_model(self, model_type: ModelType) -> bool:
        """Load a model into memory.
        
        Args:
            model_type: Type of model to load
            
        Returns:
            True if model loaded successfully
        """
        if model_type in self.models:
            self.logger.info(f"Model {model_type} already loaded")
            return True
        
        if not PYTORCH_AVAILABLE:
            self.logger.warning(f"PyTorch not available, loading placeholder for {model_type}")
            self.models[model_type] = {"type": model_type, "placeholder": True}
            return True
        
        try:
            self.logger.info(f"Loading model: {model_type}")
            
            if model_type == ModelType.INPAINT:
                self._load_inpaint_model()
            elif model_type == ModelType.DEPTH:
                self._load_depth_model()
            elif model_type == ModelType.U2NET:
                self._load_u2net_model()
            elif model_type == ModelType.SKY_SEGMENTATION:
                self._load_sky_segmentation_model()
            elif model_type == ModelType.SAM:
                self._load_sam_model()
            
            self.logger.info(f"Successfully loaded {model_type}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load {model_type}: {e}")
            return False
    
    def _load_inpaint_model(self):
        """Load Stable Diffusion Inpaint model."""
        model_id = "runwayml/stable-diffusion-inpainting"
        self.logger.info(f"Loading inpainting model from {model_id}")
        
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        )
        
        # Disable NSFW safety checker to avoid false positives blocking legitimate content
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        
        pipeline = pipeline.to(self.device)
        
        self.models[ModelType.INPAINT] = {
            "type": ModelType.INPAINT,
            "pipeline": pipeline,
        }
    
    def _load_depth_model(self):
        """Load depth estimation model (Depth Anything)."""
        model_id = "LiheYoung/depth-anything-small-hf"
        self.logger.info(f"Loading depth model from {model_id}")
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageSegmentation.from_pretrained(model_id)
        model = model.to(self.device)
        
        self.models[ModelType.DEPTH] = {
            "type": ModelType.DEPTH,
            "model": model,
            "processor": processor,
        }
    
    def _load_u2net_model(self):
        """Load U2Net foreground segmentation model."""
        # Placeholder for U2Net model loading
        self.logger.info("Loading U2Net foreground segmentation model")
        
        self.models[ModelType.U2NET] = {
            "type": ModelType.U2NET,
            "placeholder": True,
        }
    
    def _load_sky_segmentation_model(self):
        """Load sky segmentation model."""
        model_id = "mattmdjaga/segformer_b2_clothes"
        self.logger.info(f"Loading sky segmentation model from {model_id}")
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageSegmentation.from_pretrained(model_id)
        model = model.to(self.device)
        
        self.models[ModelType.SKY_SEGMENTATION] = {
            "type": ModelType.SKY_SEGMENTATION,
            "model": model,
            "processor": processor,
        }
    
    def _load_sam_model(self):
        """Load SAM (Segment Anything) model."""
        # Placeholder for SAM model loading
        self.logger.info("Loading SAM (Segment Anything) model")
        
        self.models[ModelType.SAM] = {
            "type": ModelType.SAM,
            "placeholder": True,
        }
    
    def unload_model(self, model_type: ModelType) -> bool:
        """Unload a model from memory.
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            True if model unloaded successfully
        """
        if model_type not in self.models:
            return False
        
        try:
            if PYTORCH_AVAILABLE and "pipeline" in self.models[model_type]:
                pipeline = self.models[model_type]["pipeline"]
                if hasattr(pipeline, "to"):
                    pipeline.to("cpu")
            
            self.logger.info(f"Unloading model: {model_type}")
            del self.models[model_type]
            
            if PYTORCH_AVAILABLE and self.device == "cuda":
                torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            self.logger.error(f"Error unloading {model_type}: {e}")
            return False
    
    def is_model_loaded(self, model_type: ModelType) -> bool:
        """Check if a model is loaded.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if model is loaded
        """
        return model_type in self.models
    
    def get_loaded_models(self) -> dict[str, bool]:
        """Get dictionary of all loaded models.
        
        Returns:
            Dict mapping model names to loaded status
        """
        return {str(model_type.value): True for model_type in self.models.keys()}
    
    async def inpaint(
        self,
        image_base64: str,
        mask_base64: str,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
    ) -> Optional[str]:
        """Perform inpainting on an image.
        
        Args:
            image_base64: Base64-encoded source image
            mask_base64: Base64-encoded mask
            prompt: Inpainting prompt
            negative_prompt: Negative prompt
            seed: Random seed
            
        Returns:
            Base64-encoded result or None if failed
        """
        if not self.is_model_loaded(ModelType.INPAINT):
            self.load_model(ModelType.INPAINT)
        
        if not PYTORCH_AVAILABLE or "placeholder" in self.models[ModelType.INPAINT]:
            self.logger.warning("Using placeholder inpainting (PyTorch not available)")
            return mask_base64
        
        try:
            self.logger.info(f"Inpainting with prompt: '{prompt}' on device: {self.device}")
            
            image = base64_to_image(image_base64).convert("RGB")
            mask = base64_to_image(mask_base64).convert("L")
            
            original_size = image.size
            original_mask = mask.copy()  # Keep original mask for later compositing
            self.logger.info(f"Image size: {original_size}, Image mode: {image.mode}")
            self.logger.info(f"Mask size: {mask.size}, Mask mode: {mask.mode}")
            
            # Check mask value range
            mask_min = mask.getextrema()[0]
            mask_max = mask.getextrema()[1]
            self.logger.info(f"Mask value range: {mask_min} - {mask_max}")
            
            # Find bounding box of the mask (non-black areas = area to inpaint)
            import numpy as np
            mask_array = np.array(mask)
            mask_coords = np.where(mask_array > 128)  # White areas
            
            if len(mask_coords[0]) == 0:
                self.logger.warning("No mask area found (all black)")
                return image_to_base64(image)
            
            # Step 1: Get bounding box of mask
            min_y, max_y = mask_coords[0].min(), mask_coords[0].max()
            min_x, max_x = mask_coords[1].min(), mask_coords[1].max()
            mask_width = max_x - min_x
            mask_height = max_y - min_y
            self.logger.info(f"Step 1 - Mask bbox: {mask_width}x{mask_height}")
            
            # Step 2: Expand by 30%
            expand_w = int(mask_width * 0.3)
            expand_h = int(mask_height * 0.3)
            expanded_width = mask_width + 2 * expand_w
            expanded_height = mask_height + 2 * expand_h
            self.logger.info(f"Step 2 - Expanded: {expanded_width}x{expanded_height}")
            
            crop_left = max(0, min_x - expand_w)
            crop_top = max(0, min_y - expand_h)
            crop_right = min(original_size[0], max_x + expand_w)
            crop_bottom = min(original_size[1], max_y + expand_h)
            
            # Step 3: Make it square (take larger dimension)
            crop_size = max(crop_right - crop_left, crop_bottom - crop_top)
            crop_size = (crop_size // 64) * 64  # Ensure multiple of 64
            self.logger.info(f"Step 3 - Square size: {crop_size}x{crop_size}")
            
            # Center the square crop on mask center
            center_x = (crop_left + crop_right) // 2
            center_y = (crop_top + crop_bottom) // 2
            
            crop_left = max(0, center_x - crop_size // 2)
            crop_top = max(0, center_y - crop_size // 2)
            crop_right = crop_left + crop_size
            crop_bottom = crop_top + crop_size
            
            # Clip to image bounds
            if crop_right > original_size[0]:
                crop_right = original_size[0]
                crop_left = max(0, crop_right - crop_size)
            if crop_bottom > original_size[1]:
                crop_bottom = original_size[1]
                crop_top = max(0, crop_bottom - crop_size)
            
            self.logger.info(f"Step 3 - Final crop: ({crop_left}, {crop_top}) to ({crop_right}, {crop_bottom}) = {crop_right-crop_left}x{crop_bottom-crop_top}")
            
            # Crop image and mask
            crop_box = (crop_left, crop_top, crop_right, crop_bottom)
            image_crop = image.crop(crop_box)
            mask_crop = original_mask.crop(crop_box)
            crop_size_actual = (crop_right - crop_left, crop_bottom - crop_top)
            
            self.logger.info(f"Step 4 - Cropped: {image_crop.size}, mask: {mask_crop.size}")
            
            # Step 4: Scale down to 512x512 for AI
            ai_size = 512
            if image_crop.size[0] > ai_size or image_crop.size[1] > ai_size:
                self.logger.info(f"Step 4 - Scaling crop to {ai_size}x{ai_size} for AI")
                image_crop = image_crop.resize((ai_size, ai_size), Image.Resampling.LANCZOS)
                mask_crop = mask_crop.resize((ai_size, ai_size), Image.Resampling.LANCZOS)
            
            self.logger.info(f"Step 4 - AI input: {image_crop.size}")
            
            # Stable Diffusion expects: white (255) = inpaint area, black (0) = keep
            self.logger.info(f"Using mask as-is (white=inpaint, black=keep)")
            
            pipeline = self.models[ModelType.INPAINT]["pipeline"]
            
            if seed > 0:
                torch.manual_seed(seed)
            
            self.logger.info("Step 5 - Starting inpainting...")
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image_crop,
                    mask_image=mask_crop,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    disable_progress_bar=True,
                ).images[0]
            
            self.logger.info(f"Step 5 - Inpainting completed: {result.size}")
            
            # Save debug images
            import os
            debug_dir = "inpaint_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            image_crop.save(os.path.join(debug_dir, "01_source_512.png"))
            mask_crop.save(os.path.join(debug_dir, "02_mask_512.png"))
            result.save(os.path.join(debug_dir, "03_result_raw_512.png"))
            
            # Step 5: Scale result back to original crop size
            if result.size != crop_size_actual:
                self.logger.info(f"Step 5 - Scaling result back to {crop_size_actual}")
                result = result.resize(crop_size_actual, Image.Resampling.LANCZOS)
            
            result.save(os.path.join(debug_dir, "04_result_scaled.png"))
            
            # Get the mask for this crop region at correct size
            mask_for_blend = original_mask.crop(crop_box)
            self.logger.info(f"Mask for blend: {mask_for_blend.size}")
            
            # Save parameters
            with open(os.path.join(debug_dir, "parameters.txt"), "w") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Negative Prompt: {negative_prompt}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Original Image: {original_size}\n")
                f.write(f"Mask Bbox: {mask_width}x{mask_height}\n")
                f.write(f"Expanded: {expanded_width}x{expanded_height}\n")
                f.write(f"Crop Square: {crop_size_actual}\n")
                f.write(f"AI Input: {ai_size}x{ai_size}\n")
                f.write(f"Crop Position: ({crop_left}, {crop_top})\n")
            
            self.logger.info("Debug images saved")
            
            # Step 6: Composite result back onto original image
            self.logger.info("Step 6 - Compositing back to original...")
            result_full = base64_to_image(image_base64).convert("RGB")
            self.logger.info(f"Original full image: {result_full.size}, result crop: {result.size}, blend mask: {mask_for_blend.size}")
            result_full.paste(result, crop_box, mask_for_blend)
            result = result_full
            
            self.logger.info(f"Step 6 - Final result: {result.size}")
            
            result_b64 = image_to_base64(result)
            self.logger.info(f"Encoded result: {len(result_b64)} bytes")
            return result_b64
        except Exception as e:
            self.logger.error(f"Inpainting failed: {e}", exc_info=True)
            raise
    
    async def generate_depth_mask(self, image_base64: str) -> Optional[str]:
        """Generate depth mask from image.
        
        Args:
            image_base64: Base64-encoded image
            
        Returns:
            Base64-encoded depth mask or None if failed
        """
        if not self.is_model_loaded(ModelType.DEPTH):
            self.load_model(ModelType.DEPTH)
        
        if not PYTORCH_AVAILABLE or "placeholder" in self.models[ModelType.DEPTH]:
            self.logger.warning("Using placeholder depth estimation (PyTorch not available)")
            return image_base64
        
        try:
            self.logger.info("Generating depth mask")
            
            image = base64_to_image(image_base64).convert("RGB")
            model_data = self.models[ModelType.DEPTH]
            
            processor = model_data["processor"]
            model = model_data["model"]
            
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Convert depth to grayscale image
            depth = logits.squeeze().cpu().numpy()
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype("uint8")
            
            from PIL import Image
            depth_image = Image.fromarray(depth_normalized).resize(image.size)
            
            return image_to_base64(depth_image)
        except Exception as e:
            self.logger.error(f"Depth mask generation failed: {e}")
            raise
    
    async def generate_foreground_mask(self, image_base64: str) -> Optional[str]:
        """Generate foreground/subject mask from image.
        
        Args:
            image_base64: Base64-encoded image
            
        Returns:
            Base64-encoded foreground mask or None if failed
        """
        if not self.is_model_loaded(ModelType.U2NET):
            self.load_model(ModelType.U2NET)
        
        if not PYTORCH_AVAILABLE or "placeholder" in self.models[ModelType.U2NET]:
            self.logger.warning("Using placeholder foreground masking (PyTorch not available)")
            return image_base64
        
        try:
            self.logger.info("Generating foreground mask")
            # Placeholder implementation
            return image_base64
        except Exception as e:
            self.logger.error(f"Foreground mask generation failed: {e}")
            raise
    
    async def generate_sky_mask(self, image_base64: str) -> Optional[str]:
        """Generate sky segmentation mask from image.
        
        Args:
            image_base64: Base64-encoded image
            
        Returns:
            Base64-encoded sky mask or None if failed
        """
        if not self.is_model_loaded(ModelType.SKY_SEGMENTATION):
            self.load_model(ModelType.SKY_SEGMENTATION)
        
        if not PYTORCH_AVAILABLE or "placeholder" in self.models[ModelType.SKY_SEGMENTATION]:
            self.logger.warning("Using placeholder sky segmentation (PyTorch not available)")
            return image_base64
        
        try:
            self.logger.info("Generating sky mask")
            
            image = base64_to_image(image_base64).convert("RGB")
            model_data = self.models[ModelType.SKY_SEGMENTATION]
            
            processor = model_data["processor"]
            model = model_data["model"]
            
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Convert to mask
            from PIL import Image
            mask = (logits.argmax(dim=1).squeeze().cpu().numpy() * 255).astype("uint8")
            mask_image = Image.fromarray(mask).resize(image.size)
            
            return image_to_base64(mask_image)
        except Exception as e:
            self.logger.error(f"Sky mask generation failed: {e}")
            raise
    
    async def generate_embeddings(self, image_base64: str) -> Optional[list[float]]:
        """Generate image embeddings using CLIP.
        
        Args:
            image_base64: Base64-encoded image
            
        Returns:
            List of embedding values or None if failed
        """
        if not PYTORCH_AVAILABLE:
            self.logger.warning("PyTorch not available for embeddings")
            return [0.0] * 512
        
        try:
            self.logger.info("Generating image embeddings")
            
            from transformers import CLIPProcessor, CLIPModel
            
            # Load CLIP model
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = model.to(self.device)
            
            image = base64_to_image(image_base64).convert("RGB")
            
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                image_features = model.get_image_features(**inputs)
                embeddings = image_features.cpu().numpy().tolist()[0]
            
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            # Return empty embedding as fallback
            return [0.0] * 512


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def init_model_manager(device: str = "auto") -> ModelManager:
    """Initialize the model manager.
    
    Args:
        device: Compute device ("auto", "cuda", or "cpu")
        
    Returns:
        Initialized ModelManager instance
    """
    global _model_manager
    _model_manager = ModelManager()
    
    _model_manager.set_device(device)
    return _model_manager

