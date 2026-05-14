from dataclasses import dataclass
from typing import List
import akshare as ak
import pandas as pd
from utils.cache import cached


@dataclass
class NewsItem:
    title: str
    source: str
    time: str
    summary: str
    sentiment: str      # "利好" / "利空" / "中性"
    impact_sector: str  # 影响的板块


@cached(ttl_seconds=3600)
def fetch_market_news() -> pd.DataFrame:
    """获取近期 A 股重要新闻（最近 20 条）"""
    try:
        df = ak.stock_news_em()
        return df.head(20)
    except Exception:
        try:
            df = ak.stock_info_global_em()
            return df.head(20) if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()


def format_news_for_ai(news_df: pd.DataFrame) -> str:
    """将新闻 DataFrame 格式化为 AI prompt 文本"""
    if news_df.empty:
        return "暂无新闻数据"

    lines = []
    for _, row in news_df.iterrows():
        title = row.get("title", row.get("标题", ""))
        content = row.get("content", row.get("内容", ""))
        time_str = row.get("datetime", row.get("发布时间", ""))
        lines.append(f"- [{time_str}] {title}: {str(content)[:200]}")

    return "\n".join(lines)
