from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.digimon import router as digimon_router
from app.core.config import get_settings
from app.core.logging import setup_logging
import logging

# Setup logging
settings = get_settings()
setup_logging(settings.log_level)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Digimon RAG Assistant",
    description="A RAG-powered assistant for Digimon knowledge using OpenAI, Qdrant, and MCP",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(health_router)
app.include_router(digimon_router)


@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info("Starting Digimon RAG Assistant...")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown."""
    logger.info("Shutting down Digimon RAG Assistant...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True
    )
