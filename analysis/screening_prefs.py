"""筛选偏好持久化 — 记住用户上次的筛选条件"""

import json
import os

PREFS_PATH = "screening_prefs.json"

DEFAULT_PREFS = {
    "fund_types": ["股票型", "混合型"],
    "min_return": 0.0,
    "min_scale": 1.0,
}


def load_prefs() -> dict:
    if not os.path.exists(PREFS_PATH):
        return dict(DEFAULT_PREFS)
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in DEFAULT_PREFS.items():
            if k not in data:
                data[k] = v
        return data
    except (json.JSONDecodeError, IOError):
        return dict(DEFAULT_PREFS)


def save_prefs(prefs: dict) -> None:
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(prefs, f, ensure_ascii=False, indent=2)
