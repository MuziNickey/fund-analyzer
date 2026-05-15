"""预测持久化与平滑模块

解决两大问题：
1. 预测结果每天跳变 → 指数平滑 (alpha=0.3)
2. 操作建议频繁翻转 → 趋势确认 (连续 3 天) + 操作锁定 (7 天)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

PREDICTION_PATH = "predictions_cache.json"
ALPHA = 0.3              # 新预测权重（30% 新 + 70% 旧）
CONFIRM_DAYS = 3         # 连续 N 天同方向才变更建议
LOCK_DAYS = 7            # 建议变更后锁定天数

ACTION_LABELS = {
    "hold_buy":  "持有加仓",
    "hold_wait": "暂持观望",
    "reduce":    "减仓",
    "exit":      "清仓",
}


def load_predictions() -> dict:
    """加载预测缓存，文件不存在或损坏返回空"""
    if not os.path.exists(PREDICTION_PATH):
        return {"predictions": {}, "last_refresh": ""}
    try:
        with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"predictions": {}, "last_refresh": ""}


def save_predictions(data: dict) -> None:
    """保存预测缓存"""
    with open(PREDICTION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def smooth_and_store(predictions: dict, force_refresh: bool = False) -> dict:
    """
    核心函数：对新预测做指数平滑后持久化。

    参数:
        predictions: {code: {win_prob_1m, win_prob_2m, win_prob_3m,
                             median_return_1m, ... , confidence, name}}
        force_refresh: 强制刷新（绕过平滑，直接覆盖）

    返回:
        平滑后的完整预测字典
    """
    cache = load_predictions()
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    cache["last_refresh"] = now.isoformat()

    for code, new_pred in predictions.items():
        if force_refresh or code not in cache["predictions"]:
            # 首次预测或强制刷新：直接存储
            entry = _make_entry(code, new_pred, today_str, update_count=1)
            cache["predictions"][code] = entry
            continue

        old = cache["predictions"][code]

        # 指数平滑各周期概率
        for key in ["win_prob_1m", "win_prob_2m", "win_prob_3m",
                    "median_return_1m", "median_return_2m", "median_return_3m"]:
            if key in new_pred and key in old:
                old[key] = round(
                    new_pred[key] * ALPHA + old[key] * (1 - ALPHA), 4
                )

        update_count = old.get("update_count", 0) + 1
        old["update_count"] = update_count
        old["last_updated"] = today_str
        old["confidence"] = new_pred.get("confidence", old.get("confidence", "中"))

        # ── 趋势确认逻辑 ──
        action_locked_until = old.get("action_locked_until", "")
        is_locked = action_locked_until and now.isoformat() < action_locked_until

        if not is_locked:
            win_prob_3m = old.get("win_prob_3m", 0.5)

            # 看涨判定：3 月盈利概率 >= 65%
            if win_prob_3m >= 0.65:
                old["consecutive_bullish_days"] = old.get("consecutive_bullish_days", 0) + 1
                old["consecutive_bearish_days"] = 0
            # 看跌判定：3 月盈利概率 <= 35%
            elif win_prob_3m <= 0.35:
                old["consecutive_bearish_days"] = old.get("consecutive_bearish_days", 0) + 1
                old["consecutive_bullish_days"] = 0
            else:
                old["consecutive_bullish_days"] = 0
                old["consecutive_bearish_days"] = 0

            # 连续 CONFIRM_DAYS 天确认 → 更新建议并锁定
            if old.get("consecutive_bullish_days", 0) >= CONFIRM_DAYS:
                old_action = old.get("recommended_action")
                old["recommended_action"] = "hold_buy"
                old["action_locked_until"] = (now + timedelta(days=LOCK_DAYS)).isoformat()
                if old_action != "hold_buy":
                    old["consecutive_bullish_days"] = 0

            elif old.get("consecutive_bearish_days", 0) >= CONFIRM_DAYS:
                old_action = old.get("recommended_action")
                old["recommended_action"] = "reduce"
                old["action_locked_until"] = (now + timedelta(days=LOCK_DAYS)).isoformat()
                if old_action != "reduce":
                    old["consecutive_bearish_days"] = 0

    save_predictions(cache)
    return cache


def _make_entry(code: str, pred: dict, today_str: str, update_count: int) -> dict:
    """构建缓存条目"""
    return {
        "code": code,
        "name": pred.get("name", ""),
        "win_prob_1m": pred.get("win_prob_1m", 0.5),
        "win_prob_2m": pred.get("win_prob_2m", 0.5),
        "win_prob_3m": pred.get("win_prob_3m", 0.5),
        "median_return_1m": pred.get("median_return_1m", 0.0),
        "median_return_2m": pred.get("median_return_2m", 0.0),
        "median_return_3m": pred.get("median_return_3m", 0.0),
        "confidence": pred.get("confidence", "中"),
        "last_updated": today_str,
        "update_count": update_count,
        "consecutive_bullish_days": 0,
        "consecutive_bearish_days": 0,
        "recommended_action": "hold_wait",
        "action_locked_until": "",
    }


def get_smoothed_prediction(code: str) -> Optional[dict]:
    """获取某只基金的平滑后预测"""
    cache = load_predictions()
    return cache.get("predictions", {}).get(code)


def get_long_term_suggestion(diagnosis_score: float, code: str) -> tuple:
    """
    综合诊断评分和平滑预测，给出最终建议。

    返回: (label, color_key, action_explanation)
    """
    smoothed = get_smoothed_prediction(code)
    action = smoothed.get("recommended_action", "hold_wait") if smoothed else "hold_wait"
    locked_until = smoothed.get("action_locked_until", "") if smoothed else ""
    update_count = smoothed.get("update_count", 0) if smoothed else 0

    is_locked = False
    if locked_until:
        is_locked = datetime.now().isoformat() < locked_until

    # 用平滑后概率调整评分
    adjusted_score = diagnosis_score
    if smoothed:
        prob_3m = smoothed.get("win_prob_3m", 0.5)
        # 概率偏离 50% 越远，对评分的调整越大
        adjustment = (prob_3m - 0.5) * 20
        adjusted_score = min(100, max(0, diagnosis_score + adjustment))

    if adjusted_score >= 75:
        base_label, base_color = "持有加仓", "primary"
    elif adjusted_score >= 55:
        base_label, base_color = "暂持观望", "warning"
    elif adjusted_score >= 35:
        base_label, base_color = "减仓", "danger"
    else:
        base_label, base_color = "清仓", "danger"

    # 构建说明文字
    notes = []
    if is_locked:
        unlock_date = locked_until[:10]
        notes.append(f"操作已锁定至 {unlock_date}（长线确认信号）")
    if update_count >= CONFIRM_DAYS:
        notes.append(f"已追踪 {update_count} 天，趋势信号：{ACTION_LABELS.get(action, action)}")
    if smoothed and smoothed.get("consecutive_bullish_days", 0) >= 1:
        notes.append(f"连续 {smoothed['consecutive_bullish_days']} 天看涨信号")
    if smoothed and smoothed.get("consecutive_bearish_days", 0) >= 1:
        notes.append(f"连续 {smoothed['consecutive_bearish_days']} 天看跌信号")

    return base_label, base_color, "；".join(notes) if notes else ""
