import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from app.routes import query
from app.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: just log — RAG loads lazily on first request ──────────────
    logger.info("College RAG API starting up. RAG index will load on first request.")
    yield
    logger.info("College RAG API shutting down.")

app = FastAPI(
    title="College Recommendation Assistant",
    description="Production-grade RAG system for global college advisory.",
    version="1.1.0",
    lifespan=lifespan,
    # Docs enabled — useful for testing and frontend integration
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api/v1", tags=["Assistant"])

@app.get("/", tags=["Health"])
async def root():
    return {"message": "College RAG API is running", "version": "1.1.0", "docs": "/docs"}

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": "1.1.0"}

@app.get("/test", tags=["Health"])
async def test():
    return {"status": "ok", "message": "Routes are working correctly"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG
    )
