import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.router import route_and_stream

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatMessage(BaseModel):
    user: str
    assistant: str

class AskRequest(BaseModel):
    query: str = Field(..., example="What are the fees for IIT Bombay?")
    history: Optional[List[ChatMessage]] = Field(
        default=[], description="Last 5 chat interactions"
    )

@router.post("/ask")
async def ask_question(request: AskRequest):
    """
    Streaming RAG endpoint.
    Returns a token-by-token response for real-time UI updates.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        history_list = [
            {"user": h.user, "assistant": h.assistant}
            for h in request.history
        ]
        return StreamingResponse(
            route_and_stream(request.query, history_list),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error("Streaming error for query '%s': %s", request.query, str(e))
        raise HTTPException(status_code=500, detail="Error during stream generation.")
