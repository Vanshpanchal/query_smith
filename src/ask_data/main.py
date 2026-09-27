from fastapi import FastAPI

from ask_data.config import settings

app = FastAPI(title="AskData")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {"status": "ready", "llm_model": settings.llm_model, "default_dataset": settings.default_dataset}
