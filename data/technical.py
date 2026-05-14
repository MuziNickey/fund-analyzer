import pandas as pd
import numpy as np
from typing import List, Dict


def calc_ma(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MA5, MA10, MA20 移动平均线"""
    df = df.copy()
    for period in [5, 10, 20]:
        df[f"MA{period}"] = df["单位净值"].rolling(window=period).mean()
    return df


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标：DIF, DEA, 柱状值"""
    df = df.copy()
    ema_fast = df["单位净值"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["单位净值"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    return df


def detect_cross(series_a: pd.Series, series_b: pd.Series) -> List[Dict]:
    """检测两条线的交叉点：金叉(a上穿b)和死叉(a下穿b)"""
    diff = (series_a - series_b).values
    crosses = []
    for i in range(1, len(diff)):
        if diff[i-1] <= 0 and diff[i] > 0:
            crosses.append({"index": series_a.index[i], "type": "golden_cross", "value": series_a.iloc[i]})
        elif diff[i-1] >= 0 and diff[i] < 0:
            crosses.append({"index": series_a.index[i], "type": "death_cross", "value": series_a.iloc[i]})
    return crosses


def calc_max_drawdown(nav_series: pd.Series) -> float:
    """计算最大回撤（返回负值，如 -0.15 表示最大回撤15%）"""
    cumulative_max = nav_series.cummax()
    drawdown = (nav_series - cumulative_max) / cumulative_max
    return float(drawdown.min())
