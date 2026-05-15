"""基金锁定追踪（Watchlist）模块 — 持久化"自选基金"列表，跨会话保留"""

import json
import os
from datetime import datetime

WATCHLIST_PATH = "watchlist.json"


def load_watchlist() -> dict:
    """加载锁定基金列表"""
    if not os.path.exists(WATCHLIST_PATH):
        return {"funds": {}}
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"funds": {}}


def save_watchlist(data: dict) -> None:
    """保存锁定基金列表"""
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pin_fund(code: str, name: str, source: str = "screening") -> None:
    """锁定一只基金"""
    data = load_watchlist()
    data["funds"][code] = {
        "code": code,
        "name": name,
        "pinned_date": datetime.now().strftime("%Y-%m-%d"),
        "pinned_from": source,
    }
    save_watchlist(data)


def unpin_fund(code: str) -> None:
    """解锁一只基金"""
    data = load_watchlist()
    data["funds"].pop(code, None)
    save_watchlist(data)


def is_pinned(code: str) -> bool:
    """检查基金是否已锁定"""
    data = load_watchlist()
    return code in data.get("funds", {})


def get_pinned_funds() -> list[dict]:
    """获取所有已锁定基金"""
    data = load_watchlist()
    return list(data.get("funds", {}).values())
