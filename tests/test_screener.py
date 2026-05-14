import pandas as pd
import pytest
from analysis.screener import score_funds, filter_and_rank, FundScore


@pytest.fixture
def sample_rankings():
    return pd.DataFrame({
        "基金代码": ["A001", "A002", "A003"],
        "基金简称": ["基金A", "基金B", "基金C"],
        "基金类型": ["混合型", "股票型", "指数型"],
        "近1月": ["5.2%", "3.1%", "1.8%"],
        "近3月": ["12.5%", "8.0%", "4.2%"],
        "基金规模": ["50.2亿", "10.5亿", "100.0亿"],
        "日增长率": ["0.5%", "-0.3%", "0.1%"],
    })


def test_score_funds_adds_score_column(sample_rankings):
    result = score_funds(sample_rankings)
    assert "score" in result.columns
    assert len(result) == 3
    # Scores should be between 0 and 100
    assert result["score"].between(0, 100).all()


def test_filter_and_rank_filters_by_scale(sample_rankings):
    result = filter_and_rank(
        sample_rankings,
        fund_types=["混合型", "股票型", "指数型"],
        min_return_1m=None,
        min_scale=20.0
    )
    assert len(result) == 2
    assert "A002" not in result["基金代码"].values


def test_filter_and_rank_sorts_by_score_desc(sample_rankings):
    result = filter_and_rank(
        sample_rankings,
        fund_types=["混合型", "股票型", "指数型"],
        min_return_1m=None,
        min_scale=0
    )
    scores = result["score"].values
    assert scores[0] >= scores[-1]


def test_filter_by_fund_type(sample_rankings):
    result = filter_and_rank(
        sample_rankings,
        fund_types=["混合型"],
        min_return_1m=None,
        min_scale=0
    )
    assert len(result) == 1
    assert result["基金代码"].values[0] == "A001"


def test_fund_score_dataclass():
    fs = FundScore("000001", "测试", "股票型", 0.05, 0.15, 50.0, 85.0, True)
    assert fs.code == "000001"
    assert fs.score == 85.0
    assert fs.is_recommended == True
