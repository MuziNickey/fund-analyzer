from dataclasses import dataclass
from typing import Optional
import pandas as pd
from data.fund_fetcher import enrich_fund_data


@dataclass
class FundScore:
    code: str
    name: str
    fund_type: str
    return_1m: float
    return_3m: float
    scale: float
    score: float
    is_recommended: bool = False

    def __post_init__(self):
        self.is_recommended = self.score >= 75


def _parse_pct(val) -> float:
    """解析带%的字符串为浮点数"""
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).replace("%", "")) / 100


def _parse_scale(val) -> float:
    """解析规模字符串为亿为单位的浮点数"""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("亿", "").replace("万", "")
    result = float(s)
    if "万" in str(val):
        result = result / 10000
    return result


def score_funds(df: pd.DataFrame) -> pd.DataFrame:
    """对基金 DataFrame 进行综合评分"""
    df = df.copy()

    ret_1m = df["近1月"].apply(_parse_pct)
    ret_3m = df["近3月"].apply(_parse_pct)
    daily_vol = df["日增长率"].apply(_parse_pct).abs()
    scale = df["基金规模"].apply(_parse_scale)

    # Normalize to 0-1 range
    def safe_norm(series, reverse=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        normed = (series - mn) / (mx - mn)
        return 1 - normed if reverse else normed

    ret_1m_norm = safe_norm(ret_1m)
    ret_3m_norm = safe_norm(ret_3m)
    vol_norm = safe_norm(daily_vol, reverse=True)
    scale_norm = safe_norm(scale)

    df["score"] = (
        ret_1m_norm * 0.30 +
        ret_3m_norm * 0.15 +
        vol_norm * 0.35 +
        scale_norm * 0.20
    ) * 100

    df["score"] = df["score"].round(1)
    return df


def filter_and_rank(
    df: pd.DataFrame,
    fund_types: list[str],
    min_return_1m: Optional[float],
    min_scale: float = 1.0
) -> pd.DataFrame:
    """筛选并排序基金"""
    result = enrich_fund_data(df)

    if fund_types:
        result = result[result["基金类型"].apply(
            lambda t: any(ft in str(t) for ft in fund_types)
        )]

    result = result[result["基金规模"].apply(_parse_scale) >= min_scale]

    if min_return_1m is not None:
        result = result[result["近1月"].apply(_parse_pct) >= min_return_1m / 100]

    result = score_funds(result)
    result = result.sort_values("score", ascending=False).head(10)

    return result
