import pytest
import pandas as pd
import numpy as np
from analysis.predictor import (
    PeriodPrediction,
    PredictionResult,
    build_feature_matrix,
    find_similar_patterns,
    compute_period_prediction,
    predict_fund_return,
)


def test_period_prediction_dataclass():
    pp = PeriodPrediction(
        win_probability=0.72,
        median_return=0.052,
        p25_return=0.018,
        p75_return=0.089,
        max_gain=0.15,
        max_loss=-0.08,
    )
    assert pp.win_probability == 0.72
    assert 0 <= pp.win_probability <= 1
    assert pp.median_return == 0.052
    assert pp.p25_return <= pp.median_return <= pp.p75_return


def test_prediction_result_dataclass():
    pp = PeriodPrediction(0.7, 0.05, 0.01, 0.08, 0.12, -0.05)
    pr = PredictionResult(
        code="000001",
        name="Test Fund",
        current_features={"F1_ma_align": 0.6},
        match_count=30,
        avg_similarity=0.83,
        pred_1m=pp,
        pred_2m=pp,
        pred_3m=pp,
        confidence="高",
    )
    assert pr.code == "000001"
    assert pr.confidence == "高"
    assert pr.pred_1m.win_probability == 0.7


def test_prediction_result_with_none_periods():
    pr = PredictionResult(
        code="000001",
        name="New Fund",
        current_features={"F1_ma_align": 0.5},
        match_count=0,
        avg_similarity=0.0,
        pred_1m=None,
        pred_2m=None,
        pred_3m=None,
        confidence="Low",
    )
    assert pr.pred_1m is None
    assert pr.match_count == 0


def test_build_feature_matrix_shape():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=120, freq="B")
    fund_nav = pd.Series(1.0 + np.cumsum(np.random.randn(120) * 0.005))
    bench_nav = pd.Series(1.0 + np.cumsum(np.random.randn(120) * 0.004))

    matrix = build_feature_matrix(fund_nav, bench_nav, lookback=20)

    assert matrix.shape[0] >= 30
    assert matrix.shape[1] >= 9
    assert "F1_ma_align" in matrix.columns
    assert "F2_macd_momentum" in matrix.columns
    assert "forward_22d" in matrix.columns
    assert "forward_44d" in matrix.columns
    assert "forward_66d" in matrix.columns


def test_find_similar_patterns_returns_top_n():
    np.random.seed(42)
    feature_cols = ["F1_ma_align", "F2_macd_momentum", "F3_rsi", "F4_bollinger", "F5_volatility", "F6_benchmark"]
    matrix_data = np.random.rand(10, 6)
    forward_cols = {
        "forward_22d": np.random.randn(10) * 0.1,
        "forward_44d": np.random.randn(10) * 0.1,
        "forward_66d": np.random.randn(10) * 0.1,
    }
    feat_matrix = pd.DataFrame(matrix_data, columns=feature_cols)
    for k, v in forward_cols.items():
        feat_matrix[k] = v

    current = feat_matrix.iloc[0][feature_cols].to_dict()
    result = find_similar_patterns(current, feat_matrix, top_n=5)

    assert len(result) == 5
    assert "similarity" in result.columns
    assert result.iloc[0]["similarity"] >= result.iloc[-1]["similarity"]
    assert result["similarity"].max() >= 0.99


def test_compute_period_prediction():
    forward_returns = pd.Series([0.05, 0.03, 0.08, -0.02, 0.01, 0.06, -0.04, 0.02, 0.09, 0.0])
    result = compute_period_prediction(forward_returns)
    assert isinstance(result, PeriodPrediction)
    assert result.win_probability == 0.7
    assert result.max_gain == 0.09
    assert result.max_loss == -0.04
    assert result.p25_return <= result.median_return <= result.p75_return


def test_compute_period_prediction_empty():
    assert compute_period_prediction(pd.Series(dtype=float)) is None


def test_predict_fund_return_with_synthetic_data():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    fund_nav = pd.Series(1.0 + np.cumsum(np.random.randn(200) * 0.005))

    result = predict_fund_return(fund_nav, fund_nav.copy(), code="TEST", name="Test")

    assert isinstance(result, PredictionResult)
    assert result.code == "TEST"
    assert result.match_count >= 5
    assert 0.0 <= result.avg_similarity <= 1.0
    assert result.pred_1m is not None
    assert 0 <= result.pred_1m.win_probability <= 1
    assert result.pred_2m is not None
    assert result.pred_3m is not None
    assert result.confidence in ("高", "中", "低")


def test_predict_fund_return_insufficient_data():
    np.random.seed(42)
    short_nav = pd.Series(1.0 + np.cumsum(np.random.randn(30) * 0.005))

    result = predict_fund_return(short_nav, short_nav.copy(), code="NEW", name="New Fund")

    assert result.code == "NEW"
    assert result.pred_1m is None
    assert result.pred_2m is None
    assert result.pred_3m is None
    assert result.match_count == 0
    assert result.confidence == "低"
