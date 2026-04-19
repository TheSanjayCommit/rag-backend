#!/bin/bash

echo "🚀 Starting College RAG Deployment Pipeline..."

# 1. Build the database index in an isolated process to save RAM
echo "🏗️ Step 1: Building/Checking RAG Database..."
python -c "from app.utils.indexer import build_index_if_missing; build_index_if_missing()"

# 2. Start the API server with optimized memory settings
echo "🌐 Step 2: Launching FastAPI Server..."
# Using 1 worker and low timeout to keep memory footprint minimal
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 5
