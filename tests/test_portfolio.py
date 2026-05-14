import json
import pytest
from analysis.portfolio import (
    load_portfolio,
    save_portfolio,
    add_holding,
    remove_holding,
    calc_portfolio_summary,
    diagnose_holding,
    PORTFOLIO_PATH,
)


@pytest.fixture
def temp_portfolio_file(tmp_path):
    pfile = tmp_path / "portfolio.json"
    pfile.write_text('{"holdings": [], "last_updated": ""}', encoding="utf-8")
    return str(pfile)


def test_add_holding(temp_portfolio_file):
    add_holding(temp_portfolio_file, "000001", "测试基金A", 1.5000, 10000)
    data = load_portfolio(temp_portfolio_file)
    assert len(data["holdings"]) == 1
    assert data["holdings"][0]["code"] == "000001"
    assert data["holdings"][0]["cost_nav"] == 1.5
    assert data["holdings"][0]["amount"] == 10000


def test_add_holding_dedup(temp_portfolio_file):
    add_holding(temp_portfolio_file, "000001", "基金A", 1.50, 10000)
    add_holding(temp_portfolio_file, "000001", "基金A新", 1.60, 20000)
    data = load_portfolio(temp_portfolio_file)
    assert len(data["holdings"]) == 1
    assert data["holdings"][0]["name"] == "基金A新"


def test_remove_holding(temp_portfolio_file):
    add_holding(temp_portfolio_file, "000001", "测试基金A", 1.5000, 10000)
    remove_holding(temp_portfolio_file, "000001")
    data = load_portfolio(temp_portfolio_file)
    assert len(data["holdings"]) == 0


def test_calc_portfolio_summary(temp_portfolio_file):
    add_holding(temp_portfolio_file, "000001", "基金A", 1.50, 15000)
    add_holding(temp_portfolio_file, "000002", "基金B", 2.00, 10000)
    # 基金A: 10000份 * 1.80 = 18000, 基金B: 5000份 * 1.80 = 9000
    summary = calc_portfolio_summary(
        temp_portfolio_file,
        {"000001": 1.80, "000002": 1.80}
    )
    assert summary["total_invested"] == 25000
    assert summary["total_value"] == 27000
    assert summary["total_pnl"] == 2000
    assert summary["total_pnl_pct"] == 8.0
    assert summary["holding_count"] == 2


def test_diagnose_holding_returns_score_between_0_and_100():
    result = diagnose_holding(
        nav_current=1.80,
        nav_cost=1.50,
        return_1m=0.05,
        benchmark_return=0.03,
        ma10=1.75,
        ma20=1.70,
        max_drawdown_1m=-0.03,
    )
    assert 0 <= result["score"] <= 100
    assert "suggestion" in result
    assert "label" in result
    assert "breakdown" in result
    assert result["pnl_pct"] == 20.0  # (1.80-1.50)/1.50 = 20%


def test_diagnose_holding_bullish():
    """Strong fund: bullish trend, good return"""
    result = diagnose_holding(
        nav_current=2.00, nav_cost=1.50, return_1m=0.10,
        benchmark_return=0.02, ma10=1.90, ma20=1.80,
        max_drawdown_1m=-0.01,
    )
    assert result["score"] >= 75
    assert "加仓" in result["suggestion"]


def test_diagnose_holding_bearish():
    """Weak fund: bearish trend, poor return"""
    result = diagnose_holding(
        nav_current=1.20, nav_cost=1.50, return_1m=-0.10,
        benchmark_return=0.02, ma10=1.30, ma20=1.35,
        max_drawdown_1m=-0.15,
    )
    assert result["score"] < 55
