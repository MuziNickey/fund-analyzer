# tests/test_fund_fetcher.py
import pandas as pd
from data.fund_fetcher import (
    fetch_fund_rankings,
    fetch_fund_nav_history,
    get_fund_name,
    classify_fund_type,
    enrich_fund_data,
    FundInfo
)


def test_fetch_fund_rankings_returns_dataframe():
    df = fetch_fund_rankings()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Check for expected columns (names may vary slightly with akshare versions)
    assert len(df.columns) > 5, f"Expected many columns, got {len(df.columns)}"


def test_fetch_fund_nav_history_returns_dataframe_with_data():
    df = fetch_fund_nav_history("000001")
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 20  # At least 20 trading days
    assert "净值日期" in df.columns
    assert "单位净值" in df.columns


def test_get_fund_name_returns_string():
    name = get_fund_name("000001")
    assert isinstance(name, str)
    assert len(name) > 0


def test_fund_info_dataclass():
    fi = FundInfo(
        code="000001", name="测试", fund_type="股票型",
        nav=1.5, daily_change=0.01, return_1m=0.05,
        return_3m=0.15, fund_size=50.0, sharpe=1.2
    )
    assert fi.code == "000001"
    assert fi.nav == 1.5


def test_classify_fund_type_etf():
    assert classify_fund_type("华泰柏瑞沪深300ETF") == "指数型"
    assert classify_fund_type("易方达上证50指数A") == "指数型"
    assert classify_fund_type("华夏沪深300ETF联接A") == "指数型"


def test_classify_fund_type_mixed():
    assert classify_fund_type("信澳业绩驱动混合A") == "混合型"
    assert classify_fund_type("广发稳健增长混合") == "混合型"


def test_classify_fund_type_bond():
    assert classify_fund_type("招商产业债券A") == "债券型"
    assert classify_fund_type("易方达纯债债券") == "债券型"


def test_classify_fund_type_qdii():
    assert classify_fund_type("广发纳斯达克100QDII") == "QDII"


def test_classify_fund_type_money():
    assert classify_fund_type("天弘余额宝货币") == "货币型"


def test_classify_fund_type_default_stock():
    assert classify_fund_type("华夏成长") == "股票型"
    assert classify_fund_type("xxxx") == "股票型"


def test_classify_fund_type_none():
    assert classify_fund_type(None) == "其他"
    assert classify_fund_type(123) == "其他"


def test_enrich_fund_data_adds_type_column():
    df = pd.DataFrame({
        "基金代码": ["A001", "A002"],
        "基金简称": ["测试混合A", "测试ETF联接"],
    })
    result = enrich_fund_data(df)
    assert "基金类型" in result.columns
    assert result["基金类型"].iloc[0] == "混合型"
    assert result["基金类型"].iloc[1] == "指数型"


def test_enrich_fund_data_preserves_existing_columns():
    df = pd.DataFrame({
        "基金代码": ["A001"],
        "基金简称": ["测试"],
        "基金类型": ["自定义类型"],
        "基金规模": [100.0],
    })
    result = enrich_fund_data(df)
    assert result["基金类型"].iloc[0] == "自定义类型"
    assert result["基金规模"].iloc[0] == 100.0
