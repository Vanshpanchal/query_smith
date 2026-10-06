"""API key auth + Redis fixed-window rate limit (layer 6 of the defence in depth)."""

import hashlib
import logging
import secrets
import time

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from ask_data.config import settings

log = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _matches(key: str | None, valid: set[str]) -> bool:
    return bool(key) and any(secrets.compare_digest(key, v) for v in valid)


class RateLimiter:
    """`limit` requests per `window` seconds per key. Fails open if Redis is down (logged;
    /ready reports it) so a cache outage doesn't take the API down."""

    def __init__(self, redis, limit: int, window: int = 60):
        self.redis, self.limit, self.window = redis, limit, window

    async def retry_after(self, api_key: str) -> int | None:
        """Count this request; return seconds to wait if over the limit, else None."""
        now = time.time()
        bucket = int(now // self.window)
        key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]  # never store raw keys
        redis_key = f"ratelimit:{key_id}:{bucket}"
        try:
            count = await self.redis.incr(redis_key)
            if count == 1:
                await self.redis.expire(redis_key, self.window)
        except Exception as e:  # noqa: BLE001
            log.warning("rate limiter unavailable, allowing request: %s", e)
            return None
        if count > self.limit:
            return max(1, int(self.window - now % self.window))
        return None


async def require_api_key(request: Request, key: str | None = Depends(api_key_header)) -> str:
    if not _matches(key, settings.api_key_set):
        raise HTTPException(401, "Missing or invalid X-API-Key")
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        wait = await limiter.retry_after(key)
        if wait is not None:
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(wait)})
    return key


async def require_admin_key(key: str | None = Depends(api_key_header)) -> str:
    if not _matches(key, {settings.admin_api_key} if settings.admin_api_key else set()):
        raise HTTPException(401, "Missing or invalid admin X-API-Key")
    return key
