"""Entry point for running the AI Service."""

import sys

import uvicorn

from ai_service.config import settings


def main():
    """Run the AI Service."""
    uvicorn.run(
        "ai_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
        timeout_graceful_shutdown=5,  # Add this line
    )

if __name__ == "__main__":
    main()
