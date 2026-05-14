import pytest
import pandas as pd
import numpy as np
from data.technical import calc_ma, calc_macd, detect_cross, calc_max_drawdown


@pytest.fixture
def sample_nav_df():
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    nav = [1.0 + i * 0.005 + np.sin(i * 0.1) * 0.05 for i in range(60)]
    return pd.DataFrame({"净值日期": dates, "单位净值": nav})


def test_calc_ma_adds_correct_columns(sample_nav_df):
    result = calc_ma(sample_nav_df.copy())
    assert "MA5" in result.columns
    assert "MA10" in result.columns
    assert "MA20" in result.columns


def test_calc_ma_values_correct():
    df = pd.DataFrame({"单位净值": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = calc_ma(df)
    assert result["MA5"].iloc[-1] == 3.0


def test_calc_macd_adds_dif_dea_histogram(sample_nav_df):
    result = calc_macd(sample_nav_df.copy())
    assert "DIF" in result.columns
    assert "DEA" in result.columns
    assert "MACD" in result.columns


def test_detect_cross_finds_golden_and_death_cross():
    dates = pd.date_range("2026-01-01", periods=9, freq="D")
    series_a = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    series_b = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], index=dates)
    crosses = detect_cross(series_a, series_b)
    assert len(crosses) == 2
    assert crosses[0]["index"] == dates[3]
    assert crosses[0]["type"] == "death_cross"
    assert crosses[1]["index"] == dates[7]
    assert crosses[1]["type"] == "golden_cross"


def test_calc_max_drawdown_returns_negative_float():
    nav = pd.Series([1.0, 1.2, 0.8, 0.9, 1.1])
    mdd = calc_max_drawdown(nav)
    assert isinstance(mdd, float)
    assert mdd < 0
