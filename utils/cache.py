# utils/cache.py
import time
from functools import wraps


def cached(ttl_seconds: int = 3600):
    """基于内存的 TTL 缓存装饰器"""
    def decorator(func):
        _cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in _cache and now < _cache[key]["expires_at"]:
                return _cache[key]["value"]
            result = func(*args, **kwargs)
            _cache[key] = {"value": result, "expires_at": now + ttl_seconds}
            return result

        def invalidate():
            _cache.clear()

        wrapper.invalidate = invalidate
        return wrapper
    return decorator
