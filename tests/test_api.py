"""Tests for the AI Service."""

import pytest
from fastapi.testclient import TestClient

from ai_service.main import app, app_state


@pytest.fixture
def client():
    """Create a test client."""
    # Ensure service is marked as ready for tests
    app_state.models_ready = True
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_models_status(self, client):
        """Test models status endpoint."""
        response = client.get("/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "device" in data


class TestInpaintingEndpoints:
    """Test inpainting endpoints."""
    
    def test_middleware_inpaint(self, client):
        """Test middleware-style inpainting endpoint."""
        request_data = {
            "source_id": "test_hash",
            "prompt": "test prompt",
            "negative_prompt": "avoid this",
            "mask_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "seed": 0,
        }
        response = client.post("/inpaint", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "x" in data
        assert "y" in data
        assert "color" in data
    
    def test_cloud_inpaint(self, client):
        """Test cloud-style inpainting endpoint."""
        request_data = {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "mask_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "prompt": "test prompt",
            "seed": 0,
        }
        response = client.post("/inpaint/cloud", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "color" in data


class TestMaskEndpoints:
    """Test mask generation endpoints."""
    
    def test_foreground_mask(self, client):
        """Test foreground mask generation."""
        request_data = {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        }
        response = client.post("/masks/foreground", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "mask_base64" in data
    
    def test_sky_mask(self, client):
        """Test sky mask generation."""
        request_data = {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        }
        response = client.post("/masks/sky", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "mask_base64" in data
    
    def test_depth_mask(self, client):
        """Test depth mask generation."""
        request_data = {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        }
        response = client.post("/masks/depth", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "mask_base64" in data
    
    def test_subject_mask(self, client):
        """Test subject mask generation."""
        request_data = {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        }
        response = client.post("/masks/subject", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "mask_base64" in data


class TestEmbeddingsEndpoint:
    """Test embeddings generation endpoint."""
    
    def test_embeddings(self, client):
        """Test embeddings generation."""
        request_data = {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        }
        response = client.post("/embeddings", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert isinstance(data["embeddings"], list)


class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_required_field(self, client):
        """Test handling of missing required fields."""
        response = client.post("/masks/foreground", json={})
        assert response.status_code == 500
    
    def test_service_unavailable(self):
        """Test service unavailable response."""
        # Create a client with service not ready
        app_state.models_ready = False
        app_state.error_message = "Test error"
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 503
        
        # Restore state
        app_state.models_ready = True
        app_state.error_message = None
