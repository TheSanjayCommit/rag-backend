#!/bin/bash
# Render.com start script for the College RAG Backend
# This file is used by Render to start the production server

echo "🚀 Starting College RAG API server..."
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
