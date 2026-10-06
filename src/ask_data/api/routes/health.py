from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ask_data.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """Postgres (app + reader roles) and Redis reachable → 200, else 503 with what failed."""
    state = request.app.state
    checks: dict[str, str] = {}
    for name, pool in (("postgres_app", state.app_pool), ("postgres_reader", state.reader_pool)):
        try:
            async with pool.connection(timeout=2) as conn:
                await conn.execute("SELECT 1")
            checks[name] = "ok"
        except Exception as e:  # noqa: BLE001
            checks[name] = f"error: {type(e).__name__}"
    try:
        await state.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["redis"] = f"error: {type(e).__name__}"

    ok = all(v == "ok" for v in checks.values())
    body = {
        "status": "ready" if ok else "unavailable",
        "checks": checks,
        "llm_model": settings.llm_model,
        "llm": state.llm_status,
        "datasets": [ds.name for ds in state.deps.registry.all()],
    }
    return JSONResponse(body, status_code=200 if ok else 503)
