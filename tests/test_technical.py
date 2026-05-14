import pytest
import pandas as pd
import numpy as np
from data.technical import calc_ma, calc_macd, detect_cross, calc_max_drawdown, calc_rsi, calc_bollinger, calc_volatility


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


def test_calc_rsi_value_range():
    """RSI value range must be in [0, 100]"""
    import pandas as pd
    import numpy as np
    from data.technical import calc_rsi

    np.random.seed(42)
    nav = pd.Series(1.0 + np.cumsum(np.random.randn(100) * 0.01))
    rsi = calc_rsi(nav, period=14)
    assert 0 <= rsi.iloc[-1] <= 100
    assert rsi.isna().sum() == 14  # first diff NaN + 13 rolling NaN = 14


def test_calc_bollinger_band_order():
    """Bollinger Bands: upper > middle > lower"""
    import pandas as pd
    import numpy as np
    from data.technical import calc_bollinger

    np.random.seed(42)
    nav = pd.Series(1.0 + np.cumsum(np.random.randn(30) * 0.01))
    upper, middle, lower = calc_bollinger(nav, period=20, std=2)
    assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]
    assert len(upper) == 30


def test_calc_volatility_positive_or_zero():
    """Volatility must be >= 0"""
    import pandas as pd
    from data.technical import calc_volatility

    # constant series -> vol = 0
    nav_flat = pd.Series([1.0] * 30)
    vol_flat = calc_volatility(nav_flat, period=20)
    assert vol_flat == 0.0

    # fluctuating series -> vol > 0
    import numpy as np
    np.random.seed(42)
    nav_vol = pd.Series(1.0 + np.cumsum(np.random.randn(50) * 0.01))
    vol = calc_volatility(nav_vol, period=20)
    assert vol > 0
    assert isinstance(vol, float)


def test_extract_feature_vector_six_dims_all_in_range():
    """6-dim feature vector each value in [0,1] range"""
    import pandas as pd
    import numpy as np
    from data.technical import extract_feature_vector

    np.random.seed(42)
    fund_nav = pd.Series(1.0 + np.cumsum(np.random.randn(60) * 0.01))
    bench_nav = pd.Series(1.0 + np.cumsum(np.random.randn(60) * 0.008))

    features = extract_feature_vector(fund_nav, bench_nav)

    assert len(features) == 6
    for key, val in features.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val}, expected [0,1]"
    expected_keys = {"F1_ma_align", "F2_macd_momentum", "F3_rsi", "F4_bollinger", "F5_volatility", "F6_benchmark"}
    assert set(features.keys()) == expected_keys
