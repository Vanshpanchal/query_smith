from fastapi import APIRouter, Depends, HTTPException, Request

from ask_data.api.security import require_admin_key, require_api_key
from ask_data.config import settings
from ask_data.datasets.registry import DatasetRegistry

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)], tags=["datasets"])
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)], tags=["admin"])


@router.get("/datasets")
async def list_datasets(request: Request) -> list[dict]:
    return [ds.summary() for ds in request.app.state.deps.registry.all()]


@router.get("/datasets/{name}/schema")
async def dataset_schema(name: str, request: Request) -> dict:
    registry = request.app.state.deps.registry
    if name not in registry:
        raise HTTPException(404, f"Dataset {name!r} is not loaded")
    ds = registry.get(name)
    return {
        "name": ds.name,
        "schema": ds.schema,
        "tables": [t.as_dict() for t in ds.schema_info.tables.values()],
    }


@admin_router.post("/datasets/reload")
async def reload_datasets(request: Request) -> dict:
    """Re-read datasets/*/ + app.datasets after a `make load`, without restarting."""
    async with request.app.state.app_pool.connection() as conn:
        registry = await DatasetRegistry.load(settings.datasets_dir, conn)
    request.app.state.deps.registry = registry
    return {"datasets": [ds.name for ds in registry.all()]}
