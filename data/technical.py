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


def extract_feature_vector(fund_nav: pd.Series, benchmark_nav: pd.Series) -> dict:
    """Extract 6-dim normalized feature vector from fund NAV and benchmark NAV"""

    def _sigmoid(x: float, scale: float = 1.0) -> float:
        return float(1.0 / (1.0 + np.exp(-x * scale)))

    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    nav = fund_nav.values
    ma5_series = pd.Series(nav).rolling(5).mean()
    ma10_series = pd.Series(nav).rolling(10).mean()
    ma20_series = pd.Series(nav).rolling(20).mean()
    rsi = calc_rsi(pd.Series(nav), 14)
    upper, middle, lower = calc_bollinger(pd.Series(nav), 20, 2.0)
    vol = calc_volatility(pd.Series(nav), 20)

    ema12 = pd.Series(nav).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(nav).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()

    idx = -1
    nav_last = nav[idx]
    eps = 1e-10

    f1 = _sigmoid(
        ((nav_last - ma5_series.iloc[idx]) / (ma5_series.iloc[idx] + eps)
         + (ma5_series.iloc[idx] - ma10_series.iloc[idx]) / (ma10_series.iloc[idx] + eps)
         + (ma10_series.iloc[idx] - ma20_series.iloc[idx]) / (ma20_series.iloc[idx] + eps)) / 3,
        10.0
    )

    f2 = _sigmoid((dif.iloc[idx] - dea.iloc[idx]) / (nav_last + eps) * 100, 1.0)

    rsi_val = rsi.iloc[idx]
    f3 = _clip01((rsi_val - 30.0) / 40.0) if not np.isnan(rsi_val) else 0.5

    band_range = upper.iloc[idx] - lower.iloc[idx]
    f4 = _clip01((nav_last - lower.iloc[idx]) / (band_range + eps)) if band_range > eps else 0.5

    f5 = _clip01(vol / 0.5)

    fund_ret = (nav_last - nav[max(0, idx - 20)]) / (nav[max(0, idx - 20)] + eps)
    bench_vals = benchmark_nav.values
    bench_ret = (bench_vals[idx] - bench_vals[max(0, idx - 20)]) / (bench_vals[max(0, idx - 20)] + eps)
    f6 = _sigmoid((fund_ret - bench_ret) * 5, 1.0)

    return {
        "F1_ma_align": round(f1, 4),
        "F2_macd_momentum": round(f2, 4),
        "F3_rsi": round(f3, 4),
        "F4_bollinger": round(f4, 4),
        "F5_volatility": round(f5, 4),
        "F6_benchmark": round(f6, 4),
    }
