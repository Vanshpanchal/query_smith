"""Answer cache key + fail-open behaviour (no Redis needed)."""

from decimal import Decimal

from ask_data.api.cache import AnswerCache, cache_key, normalize_question


def test_normalized_questions_share_a_key():
    a = cache_key("ipl", "v1", "Who won the 2016 final?", "m")
    assert a == cache_key("ipl", "v1", "  who WON the   2016 final ", "m")
    assert normalize_question("Top 5 run scorers?!") == "top 5 run scorers"


def test_dataset_version_and_model_are_part_of_the_key():
    base = cache_key("ipl", "v1", "q", "m")
    assert base != cache_key("ipl", "v2", "q", "m")  # reloaded dataset → old answers unreachable
    assert base != cache_key("ipl", "v1", "q", "other-model")
    assert base != cache_key("sales", "v1", "q", "m")


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.data, self.fail = {}, fail

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.data.get(key)

    async def set(self, key, value, ex):
        if self.fail:
            raise ConnectionError("redis down")
        self.data[key] = value


async def test_round_trip_keeps_only_answer_fields():
    cache = AnswerCache(FakeRedis(), ttl_s=60)
    await cache.put("k", {"sql": "SELECT 1", "rows": [[Decimal("1.5")]], "messages": ["x"]})
    hit = await cache.get("k")
    assert hit["sql"] == "SELECT 1" and hit["rows"] == [[1.5]] and "messages" not in hit


async def test_redis_down_fails_open():
    cache = AnswerCache(FakeRedis(fail=True), ttl_s=60)
    await cache.put("k", {"sql": "SELECT 1"})  # no exception
    assert await cache.get("k") is None


async def test_ttl_zero_disables_the_cache():
    redis = FakeRedis()
    cache = AnswerCache(redis, ttl_s=0)
    await cache.put("k", {"sql": "SELECT 1"})
    assert redis.data == {} and await cache.get("k") is None
