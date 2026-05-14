# tests/test_cache.py
import time
from utils.cache import cached


def test_cached_returns_same_value_within_ttl():
    call_count = [0]

    @cached(ttl_seconds=10)
    def fetch_data():
        call_count[0] += 1
        return call_count[0]

    r1 = fetch_data()
    r2 = fetch_data()
    assert r1 == r2 == 1
    assert call_count[0] == 1


def test_cached_refreshes_after_ttl():
    call_count = [0]

    @cached(ttl_seconds=0.1)
    def fetch_data():
        call_count[0] += 1
        return call_count[0]

    r1 = fetch_data()
    time.sleep(0.15)
    r2 = fetch_data()
    assert r1 == 1
    assert r2 == 2
    assert call_count[0] == 2


def test_invalidate_clears_cache():
    call_count = [0]

    @cached(ttl_seconds=10)
    def fetch_data():
        call_count[0] += 1
        return call_count[0]

    r1 = fetch_data()
    fetch_data.invalidate()
    r2 = fetch_data()
    assert r1 == 1
    assert r2 == 2
    assert call_count[0] == 2


def test_different_args_get_different_cache_entries():
    call_log = []

    @cached(ttl_seconds=10)
    def fetch_data(code):
        call_log.append(code)
        return f"result_{code}"

    r1 = fetch_data("000001")
    r2 = fetch_data("000002")
    r3 = fetch_data("000001")  # should hit cache

    assert r1 == "result_000001"
    assert r2 == "result_000002"
    assert r3 == "result_000001"
    assert len(call_log) == 2  # only called twice (000001 once, 000002 once)
