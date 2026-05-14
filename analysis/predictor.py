"""Fund profit prediction engine — technical feature vector + historical pattern matching"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from data.technical import extract_feature_vector


@dataclass
class PeriodPrediction:
    """Prediction result for a single time horizon"""
    win_probability: float
    median_return: float
    p25_return: float
    p75_return: float
    max_gain: float
    max_loss: float


@dataclass
class PredictionResult:
    """Complete prediction result for a single fund"""
    code: str
    name: str
    current_features: dict
    match_count: int
    avg_similarity: float
    pred_1m: Optional[PeriodPrediction]
    pred_2m: Optional[PeriodPrediction]
    pred_3m: Optional[PeriodPrediction]
    confidence: str


FEATURE_COLS = ["F1_ma_align", "F2_macd_momentum", "F3_rsi", "F4_bollinger", "F5_volatility", "F6_benchmark"]


def build_feature_matrix(
    fund_nav: pd.Series,
    benchmark_nav: pd.Series,
    lookback: int = 20,
    return_horizons: tuple = (22, 44, 66),
) -> pd.DataFrame:
    """Rolling window feature extraction + forward returns for each historical snapshot"""
    rows = []
    nav_vals = fund_nav.values

    for i in range(lookback, len(nav_vals)):
        max_horizon = max(return_horizons)
        if i + max_horizon >= len(nav_vals):
            break

        fund_window = fund_nav.iloc[i - lookback:i + 1].reset_index(drop=True)
        bench_window = benchmark_nav.iloc[i - lookback:i + 1].reset_index(drop=True)

        try:
            features = extract_feature_vector(fund_window, bench_window)
        except Exception:
            continue

        row = {**features}
        for h in return_horizons:
            start_nav = nav_vals[i]
            end_nav = nav_vals[i + h]
            row[f"forward_{h}d"] = float((end_nav - start_nav) / start_nav)

        rows.append(row)

    return pd.DataFrame(rows)


def find_similar_patterns(
    current_features: dict,
    feature_matrix: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Find top_n historical windows most similar to current features via cosine similarity"""
    current_vec = np.array([[current_features[k] for k in FEATURE_COLS]])
    hist_mat = feature_matrix[FEATURE_COLS].values

    current_norm = current_vec / (np.linalg.norm(current_vec, axis=1, keepdims=True) + 1e-10)
    hist_norm = hist_mat / (np.linalg.norm(hist_mat, axis=1, keepdims=True) + 1e-10)

    similarity = (current_norm @ hist_norm.T).flatten()

    result = feature_matrix.copy()
    result["similarity"] = similarity
    return result.nlargest(top_n, "similarity")


def compute_period_prediction(forward_returns: pd.Series) -> Optional[PeriodPrediction]:
    """Compute statistical prediction from a set of historical forward returns"""
    returns = forward_returns.dropna()
    if len(returns) < 5:
        return None

    positive_mask = returns > 0
    win_prob = float(positive_mask.sum() / len(returns))

    return PeriodPrediction(
        win_probability=round(win_prob, 2),
        median_return=round(float(returns.median()), 4),
        p25_return=round(float(returns.quantile(0.25)), 4),
        p75_return=round(float(returns.quantile(0.75)), 4),
        max_gain=round(float(returns.max()), 4),
        max_loss=round(float(returns.min()), 4),
    )


def predict_fund_return(
    fund_nav: pd.Series,
    benchmark_nav: pd.Series,
    code: str = "",
    name: str = "",
    lookback: int = 20,
    top_n: int = 30,
) -> PredictionResult:
    """Main prediction function: input fund NAV and benchmark NAV, return PredictionResult"""
    if len(fund_nav) < 60:
        return PredictionResult(
            code=code, name=name,
            current_features={},
            match_count=0, avg_similarity=0.0,
            pred_1m=None, pred_2m=None, pred_3m=None,
            confidence="低",
        )

    current_features = extract_feature_vector(
        fund_nav.iloc[-lookback - 1:].reset_index(drop=True),
        benchmark_nav.iloc[-lookback - 1:].reset_index(drop=True),
    )

    feat_matrix = build_feature_matrix(fund_nav, benchmark_nav, lookback=lookback)

    if feat_matrix.empty or len(feat_matrix) < 5:
        return PredictionResult(
            code=code, name=name,
            current_features=current_features,
            match_count=0, avg_similarity=0.0,
            pred_1m=None, pred_2m=None, pred_3m=None,
            confidence="低",
        )

    matches = find_similar_patterns(current_features, feat_matrix, top_n=top_n + 1)
    if len(matches) > 1:
        matches = matches.iloc[1:]
    elif len(matches) > 0 and matches.iloc[0]["similarity"] > 0.99:
        matches = matches.iloc[1:]

    avg_sim = float(matches["similarity"].mean()) if len(matches) > 0 else 0.0

    pred_1m = compute_period_prediction(matches["forward_22d"]) if len(matches) >= 5 else None
    pred_2m = compute_period_prediction(matches["forward_44d"]) if len(matches) >= 5 else None
    pred_3m = compute_period_prediction(matches["forward_66d"]) if len(matches) >= 5 else None

    if avg_sim >= 0.75:
        confidence = "高"
    elif avg_sim >= 0.50:
        confidence = "中"
    else:
        confidence = "低"

    return PredictionResult(
        code=code, name=name,
        current_features=current_features,
        match_count=len(matches),
        avg_similarity=round(avg_sim, 4),
        pred_1m=pred_1m,
        pred_2m=pred_2m,
        pred_3m=pred_3m,
        confidence=confidence,
    )
