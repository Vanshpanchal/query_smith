from ask_data.api.security import RateLimiter


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.counts: dict[str, int] = {}
        self.fail = fail

    async def incr(self, key: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        pass


async def test_limit_per_key():
    limiter = RateLimiter(FakeRedis(), limit=2)
    assert await limiter.retry_after("a") is None
    assert await limiter.retry_after("a") is None
    wait = await limiter.retry_after("a")
    assert wait is not None and 1 <= wait <= 60
    assert await limiter.retry_after("b") is None  # other key, own window


async def test_raw_key_not_stored():
    redis = FakeRedis()
    await RateLimiter(redis, limit=1).retry_after("secret-key")
    assert not any("secret-key" in k for k in redis.counts)


async def test_fails_open_when_redis_down():
    assert await RateLimiter(FakeRedis(fail=True), limit=0).retry_after("a") is None
