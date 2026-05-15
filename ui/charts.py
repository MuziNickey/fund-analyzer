import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from ui.theme import COLORS, PLOTLY_TEMPLATE, PLOTLY_LAYOUT_DEFAULTS


def plot_nav_with_ma(
    df: pd.DataFrame,
    fund_name: str,
    cost_line: float | None = None
) -> go.Figure:
    """绘制净值走势 + MA 均线图"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["净值日期"], y=df["单位净值"],
        mode="lines", name="单位净值",
        line=dict(color=COLORS["chart_blue"], width=2)
    ))

    ma_config = [
        ("MA5", COLORS["chart_orange"], "dash"),
        ("MA10", COLORS["chart_green"], "dash"),
        ("MA20", COLORS["chart_red"], "dot"),
    ]
    for col, color, style in ma_config:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["净值日期"], y=df[col],
                mode="lines", name=col,
                line=dict(color=color, dash=style, width=1)
            ))

    if cost_line is not None:
        fig.add_hline(
            y=cost_line, line_dash="dash", line_color=COLORS["chart_red"],
            annotation_text=f"成本: {cost_line:.4f}",
            annotation_position="top right"
        )

    fig.update_layout(
        title=f"{fund_name} — 净值走势与均线",
        xaxis_title="日期",
        yaxis_title="净值",
        height=400,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def plot_macd(df: pd.DataFrame, fund_name: str) -> go.Figure:
    """绘制 MACD 柱状图"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=df["净值日期"], y=df["单位净值"],
        mode="lines", name="净值",
        line=dict(color=COLORS["chart_blue"], width=1.5)
    ), row=1, col=1)

    if "DIF" in df.columns and "DEA" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["净值日期"], y=df["DIF"],
            mode="lines", name="DIF",
            line=dict(color=COLORS["chart_orange"], width=1)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df["净值日期"], y=df["DEA"],
            mode="lines", name="DEA",
            line=dict(color=COLORS["chart_green"], width=1)
        ), row=2, col=1)

        macd_vals = df.get("MACD", pd.Series([0] * len(df)))
        colors = [COLORS["danger"] if v >= 0 else COLORS["chart_green"]
                  for v in macd_vals]
        fig.add_trace(go.Bar(
            x=df["净值日期"], y=macd_vals,
            name="MACD柱",
            marker_color=colors,
            opacity=0.6
        ), row=2, col=1)

    fig.update_layout(
        title=f"{fund_name} — MACD 指标",
        height=400,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    return fig


def plot_volume(df: pd.DataFrame, fund_name: str) -> go.Figure:
    """绘制成交量/日涨跌幅图"""
    fig = go.Figure()
    vol_col = None
    vol_label = "成交量"
    for col in ["成交量", "volume", "手"]:
        if col in df.columns:
            vol_col = col
            break

    if vol_col is None and "日增长率" in df.columns:
        vol_col = "日增长率"
        vol_label = "日涨跌幅 (%)"

    if vol_col:
        if vol_col == "日增长率":
            vals = pd.to_numeric(df[vol_col], errors="coerce")
            colors = [COLORS["danger"] if v >= 0 else COLORS["chart_green"]
                      for v in vals]
        else:
            diffs = df["单位净值"].diff()
            colors = [COLORS["danger"] if diffs.iloc[i] >= 0 else COLORS["chart_green"]
                      for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df["净值日期"], y=df[vol_col],
            name=vol_label, marker_color=colors,
            opacity=0.6
        ))
    else:
        fig.add_annotation(
            text="暂无成交量数据", showarrow=False,
            xref="paper", yref="paper", x=0.5, y=0.5
        )

    fig.update_layout(
        title=f"{fund_name} — {vol_label}",
        xaxis_title="日期",
        yaxis_title=vol_label,
        height=250,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def plot_return_distribution(pred_1m, pred_2m, pred_3m, fund_name: str) -> go.Figure:
    """Plot predicted return distribution for 3 time horizons"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("近1月", "近2月", "近3月"),
        shared_yaxes=True,
    )

    periods = [(pred_1m, 1), (pred_2m, 2), (pred_3m, 3)]
    for pred, col in periods:
        if pred is None:
            continue
        x_vals = [pred.max_loss, pred.p25_return, pred.median_return, pred.p75_return, pred.max_gain]
        x_labels = ["最大亏损", "P25", "中位", "P75", "最大收益"]
        colors_bar = [COLORS["danger"] if v < 0 else COLORS["primary"] for v in x_vals]

        fig.add_trace(
            go.Bar(
                x=x_labels, y=x_vals,
                marker_color=colors_bar,
                text=[f"{v:+.1%}" for v in x_vals],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=col,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=col)

    fig.update_layout(
        title=f"{fund_name} — 盈利预测分布",
        height=300,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=60),
        font=dict(color=COLORS["text_primary"]),
    )
    fig.update_xaxes(title_text="统计分位", row=1, col=1)
    fig.update_xaxes(title_text="统计分位", row=1, col=2)
    fig.update_xaxes(title_text="统计分位", row=1, col=3)
    fig.update_yaxes(title_text="收益率", row=1, col=1)
    fig.add_annotation(
        text="P25=偏保守 · 中位=中性预期 · P75=偏乐观 | 柱高=预期收益率 柱色=盈亏方向",
        xref="paper", yref="paper", x=0.5, y=-0.18,
        showarrow=False, font=dict(size=11, color=COLORS["text_secondary"]),
    )
    return fig
