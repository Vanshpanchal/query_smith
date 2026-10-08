"""The demo page: one static HTML file that streams the SSE events and draws the ChartSpec."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ask_data.config import settings

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

router = APIRouter(include_in_schema=False)


@router.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@router.get("/web-config")
async def web_config() -> dict:
    """DEMO_API_KEY is meant to be public (a rate-limited key for the live demo); the page
    pre-fills it so visitors can ask without one. Empty → the visitor types a key."""
    key = settings.demo_api_key.strip()
    return {"demo_api_key": key if key in settings.api_key_set else None}
