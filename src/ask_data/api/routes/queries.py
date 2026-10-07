from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ask_data.api import query_log
from ask_data.api.security import require_api_key

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)], tags=["queries"])


class FeedbackRequest(BaseModel):
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=1000)


@router.post("/queries/{query_id}/feedback")
async def feedback(query_id: UUID, body: FeedbackRequest, request: Request) -> dict:
    """👍 (1) / 👎 (-1) on an answer, by the `query_id` of its `answer` event. Sending it again
    replaces the previous rating."""
    ok = await query_log.set_feedback(
        request.app.state.app_pool, str(query_id), body.rating, body.comment
    )
    if not ok:
        raise HTTPException(404, f"Unknown query {query_id}")
    return {"query_id": str(query_id), "rating": body.rating}
