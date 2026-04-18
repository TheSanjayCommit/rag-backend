import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from app.routes import query
from app.services.rag_service import initialize_rag
from app.config.settings import settings

# Configure structured logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG indices (FAISS + Embeddings)...")
    initialize_rag()
    logger.info("RAG Ready. Application startup complete.")
    yield
    logger.info("Application shutting down. Cleaning up resources.")

app = FastAPI(
    title="College Recommendation Assistant",
    description="Production-grade RAG system for global college advisory.",
    version="1.1.0",
    lifespan=lifespan,
    # Disable docs in production
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS: Allow origins from env var in production, wildcard in development
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api/v1", tags=["Assistant"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.1.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG
    )
