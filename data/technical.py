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


def calc_rsi(nav: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI, returns 0-100 Series"""
    delta = nav.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_bollinger(nav: pd.Series, period: int = 20, std: float = 2.0):
    """Calculate Bollinger Bands, returns (upper, middle, lower) Series"""
    middle = nav.rolling(window=period).mean()
    rolling_std = nav.rolling(window=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def calc_volatility(nav: pd.Series, period: int = 20) -> float:
    """Calculate annualized volatility (last 'period' days std * sqrt(252))"""
    returns = nav.pct_change().dropna().iloc[-period:]
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * (252 ** 0.5))
