# data/fund_fetcher.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import akshare as ak
from utils.cache import cached


@dataclass
class FundInfo:
    code: str
    name: str
    fund_type: str
    nav: float           # 最新净值
    daily_change: float  # 日涨跌幅 %
    return_1m: float     # 近1月收益 %
    return_3m: float     # 近3月收益 %
    fund_size: float     # 规模 (亿)
    sharpe: float        # 夏普比率


@cached(ttl_seconds=7200)
def fetch_fund_rankings() -> pd.DataFrame:
    """获取全市场开放式基金排名，返回 DataFrame"""
    try:
        df = ak.fund_open_fund_rank_em(symbol="全部")
        return df
    except Exception:
        # Fallback: try another symbol format
        try:
            df = ak.fund_open_fund_rank_em(symbol="开放式基金")
            return df
        except Exception:
            return pd.DataFrame()


@cached(ttl_seconds=14400)
def fetch_fund_nav_history(code: str, days: int = 90) -> pd.DataFrame:
    """获取单只基金历史净值数据"""
    start_dt = datetime.now() - timedelta(days=days)

    def _post_process(df: pd.DataFrame) -> pd.DataFrame:
        """Shared post-processing: parse dates, filter, sort, clean."""
        if df is not None and not df.empty:
            df["净值日期"] = pd.to_datetime(df["净值日期"])
            df = df[df["净值日期"] >= start_dt]
            df = df.sort_values("净值日期")
            df["单位净值"] = pd.to_numeric(df["单位净值"], errors="coerce")
            df = df.dropna(subset=["单位净值"])
        return df if df is not None else pd.DataFrame()

    try:
        df = ak.fund_open_fund_info_em(
            symbol=code,
            indicator="单位净值走势"
        )
        return _post_process(df)
    except Exception:
        # Try alternate indicator name
        try:
            df = ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")
            return _post_process(df)
        except Exception:
            return pd.DataFrame()


def classify_fund_type(name: str) -> str:
    """根据基金名称推断基金类型"""
    if not isinstance(name, str):
        return "其他"
    n = name.upper()
    if "QDII" in n:
        return "QDII"
    if "ETF" in n or "指数" in name:
        return "指数型"
    if "债券" in name or "债" in name:
        return "债券型"
    if "混合" in name:
        return "混合型"
    if "货币" in name or "货" in name:
        return "货币型"
    if "LOF" in n or "FOF" in n or "基金中基金" in name:
        return "混合型"
    return "股票型"


@cached(ttl_seconds=7200)
def fetch_fund_scales() -> pd.DataFrame:
    """获取全市场基金规模数据（总募集规模 + 最近总份额）"""
    try:
        df = ak.fund_scale_open_sina()
        return df
    except Exception:
        return pd.DataFrame()


def enrich_fund_data(df: pd.DataFrame) -> pd.DataFrame:
    """为基金排名数据补充「基金类型」和「基金规模」列（仅在缺失时）"""
    df = df.copy()
    if "基金类型" not in df.columns:
        df["基金类型"] = df["基金简称"].apply(classify_fund_type)
    if "基金规模" not in df.columns:
        try:
            scales_df = fetch_fund_scales()
            if not scales_df.empty and "基金代码" in scales_df.columns:
                scale_map = {}
                for _, row in scales_df.iterrows():
                    code = str(row.get("基金代码", ""))
                    shares = row.get("最近总份额", 0)
                    try:
                        scale_map[code] = float(shares) / 1e8
                    except (ValueError, TypeError):
                        scale_map[code] = 0.0
                df["基金规模"] = df["基金代码"].map(scale_map).fillna(0.0)
            else:
                df["基金规模"] = 0.0
        except Exception:
            df["基金规模"] = 0.0
    return df


def get_fund_name(code: str) -> str:
    """根据基金代码获取名称（从排名数据中查）"""
    try:
        df = fetch_fund_rankings()
        match = df[df["基金代码"] == code]
        if not match.empty:
            return str(match.iloc[0]["基金简称"])
    except Exception:
        pass
    return code
