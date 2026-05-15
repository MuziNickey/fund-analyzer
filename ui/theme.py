"""金融科技深色主题 — 集中管理色彩、CSS 和 Plotly 模板"""

import streamlit as st

# ── 色彩常量 ──────────────────────────────────────────────
COLORS = {
    # 功能色
    "primary":       "#00D4AA",  # 青绿 — 增长/正面/盈利
    "danger":        "#FF6B6B",  # 柔和红 — 风险/下跌/亏损
    "warning":       "#FFD93D",  # 琥珀 — 中性/观望/注意
    "info":          "#2D7FF9",  # 蓝色 — 信息/链接
    "muted":         "#9E9E9E",  # 灰色 — 低置信度/禁用
    # 背景色
    "bg_main":       "#0E1117",  # 深蓝黑 — 页面主背景
    "bg_card":       "#1A1D24",  # 浅一级 — 卡片/面板/侧边栏
    "bg_elevated":   "#22262F",  # 浅二级 — hover/浮层
    # 文字色
    "text_primary":  "#E0E0E0",  # 主文字
    "text_secondary":"#888888",  # 辅助文字/说明
    # 图表色
    "chart_blue":    "#2D7FF9",
    "chart_orange":  "#FFB347",
    "chart_green":   "#00D4AA",
    "chart_red":     "#FF6B6B",
    "chart_purple":  "#BB86FC",
}

# ── 语义色映射 ────────────────────────────────────────────
SENTIMENT_COLORS = {
    "偏暖": COLORS["danger"],
    "偏冷": COLORS["chart_green"],
    "中性": COLORS["warning"],
}

LABEL_COLORS = {
    "持有加仓": COLORS["primary"],
    "暂持观望": COLORS["warning"],
    "减仓":     COLORS["danger"],
    "清仓":     COLORS["danger"],
}

# ── Plotly 配置 ───────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT_DEFAULTS = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(color=COLORS["text_secondary"], size=11),
    ),
    font=dict(color=COLORS["text_primary"]),
)

# ── CSS 样式 ──────────────────────────────────────────────
CSS = f"""
<style>
    /* ── 全局 ── */
    .stApp {{
        background: {COLORS["bg_main"]};
    }}
    h1, h2, h3, h4 {{
        color: {COLORS["text_primary"]} !important;
    }}
    h2 {{
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 0.4em;
    }}
    p, span, label, .stMarkdown {{
        color: {COLORS["text_primary"]};
    }}
    hr {{
        border-color: rgba(255,255,255,0.06);
    }}

    /* ── 侧边栏 ── */
    [data-testid="stSidebar"] {{
        background: {COLORS["bg_card"]};
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.08);
    }}
    [data-testid="stSidebar"] .stMarkdown h4 {{
        color: {COLORS["primary"]} !important;
    }}

    /* ── Tab 导航 ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS["text_secondary"]};
        font-size: 0.95rem;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background: transparent;
        transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS["text_primary"]};
        background: rgba(255,255,255,0.03);
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS["primary"]} !important;
        border-bottom: 2px solid {COLORS["primary"]};
    }}

    /* ── Expander 折叠面板 ── */
    .streamlit-expanderHeader {{
        background: {COLORS["bg_card"]};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 14px 18px;
        font-weight: 600;
        color: {COLORS["text_primary"]};
        transition: all 0.15s ease;
    }}
    .streamlit-expanderHeader:hover {{
        border-color: {COLORS["primary"]};
        background: {COLORS["bg_elevated"]};
    }}
    .streamlit-expanderHeader svg {{
        fill: {COLORS["primary"]};
    }}

    /* ── Metric 指标卡片 ── */
    [data-testid="stMetric"] {{
        background: {COLORS["bg_card"]};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 16px;
    }}
    [data-testid="stMetric"] label {{
        color: {COLORS["text_secondary"]} !important;
        font-size: 0.8rem;
    }}
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* ── DataFrame 表格 ── */
    [data-testid="stDataFrame"] {{
        border-radius: 8px;
        overflow: hidden;
    }}
    [data-testid="stDataFrame"] table {{
        border-collapse: separate;
        border-spacing: 0;
    }}

    /* ── 按钮 ── */
    .stButton > button {{
        background: {COLORS["bg_card"]};
        color: {COLORS["text_primary"]};
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {COLORS["primary"]};
        color: {COLORS["primary"]};
        background: {COLORS["bg_elevated"]};
    }}

    /* ── Form 表单 ── */
    [data-testid="stForm"] {{
        background: {COLORS["bg_card"]};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 20px;
    }}

    /* ── Input 输入框 ── */
    input, textarea, [data-baseweb="input"] {{
        background: {COLORS["bg_elevated"]} !important;
        color: {COLORS["text_primary"]} !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 6px !important;
    }}

    /* ── Slider ── */
    [data-testid="stSlider"] div[data-baseweb="slider"] div {{
        background: {COLORS["primary"]};
    }}

    /* ── Spinner ── */
    .stSpinner > div {{
        border-color: {COLORS["primary"]} !important;
    }}

    /* ── 滚动条 ── */
    ::-webkit-scrollbar {{
        width: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(255,255,255,0.08);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(255,255,255,0.15);
    }}
</style>
"""


def apply_theme():
    """注入深色主题 CSS（必须在任何页面内容之前调用）"""
    st.markdown(CSS, unsafe_allow_html=True)
