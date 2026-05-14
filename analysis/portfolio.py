import json
from datetime import datetime
from typing import Optional


PORTFOLIO_PATH = "portfolio.json"


def load_portfolio(filepath: str) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"holdings": [], "last_updated": ""}


def save_portfolio(filepath: str, data: dict):
    data_copy = {**data, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_copy, f, ensure_ascii=False, indent=2)


def add_holding(filepath: str, code: str, name: str, cost_nav: float, amount: float):
    data = load_portfolio(filepath)
    data["holdings"] = [h for h in data["holdings"] if h["code"] != code]
    data["holdings"].append({
        "code": code,
        "name": name,
        "cost_nav": cost_nav,
        "amount": amount,
        "added_date": datetime.now().strftime("%Y-%m-%d")
    })
    save_portfolio(filepath, data)


def remove_holding(filepath: str, code: str):
    data = load_portfolio(filepath)
    data["holdings"] = [h for h in data["holdings"] if h["code"] != code]
    save_portfolio(filepath, data)


def calc_portfolio_summary(filepath: str, current_navs: dict[str, float]) -> dict:
    data = load_portfolio(filepath)
    total_invested = 0
    total_value = 0

    for h in data["holdings"]:
        code = h["code"]
        nav = current_navs.get(code, h["cost_nav"])
        shares = h["amount"] / h["cost_nav"]
        total_invested += h["amount"]
        total_value += shares * nav

    pnl = total_value - total_invested
    pnl_pct = (pnl / total_invested) * 100 if total_invested > 0 else 0

    return {
        "total_invested": round(total_invested, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(pnl, 2),
        "total_pnl_pct": round(pnl_pct, 2),
        "holding_count": len(data["holdings"]),
    }


def diagnose_holding(
    nav_current: float,
    nav_cost: float,
    return_1m: float,
    benchmark_return: float,
    ma10: float,
    ma20: float,
    max_drawdown_1m: float,
) -> dict:
    """四维度持仓诊断评分"""
    if nav_cost <= 0:
        nav_cost = 1.0

    # Trend (35%): bullish alignment = nav > MA10 > MA20
    if nav_current > ma10 > ma20:
        trend_score = 90
    elif nav_current > ma10:
        trend_score = 70
    elif nav_current > ma20:
        trend_score = 55
    else:
        trend_score = 30

    # Relative strength (25%)
    relative_diff = return_1m - benchmark_return
    relative_score = min(100, max(0, 50 + relative_diff * 1000))

    # Drawdown risk (20%): shallower drawdown = better
    drawdown_score = min(100, max(0, 100 + max_drawdown_1m * 200))

    # Market match (20%)
    match_score = min(100, max(0, 50 + return_1m * 500))

    total = (
        trend_score * 0.35 +
        relative_score * 0.25 +
        drawdown_score * 0.20 +
        match_score * 0.20
    )

    if total >= 75:
        suggestion = "继续持有，可加仓"
        label = "\U0001f7e2 持有加仓"
    elif total >= 55:
        suggestion = "暂持观望，等信号明确"
        label = "\U0001f7e1 暂持观望"
    elif total >= 35:
        suggestion = "减仓 30%-50%，降低风险"
        label = "\U0001f7e0 减仓"
    else:
        suggestion = "建议清仓，及时止损"
        label = "\U0001f534 清仓"

    return {
        "score": round(total, 1),
        "suggestion": suggestion,
        "label": label,
        "breakdown": {
            "trend": round(trend_score, 1),
            "relative_strength": round(relative_score, 1),
            "drawdown_risk": round(drawdown_score, 1),
            "market_match": round(match_score, 1),
        },
        "pnl_pct": round((nav_current - nav_cost) / nav_cost * 100, 2),
    }
