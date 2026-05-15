import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="A股基金智能分析助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.theme import apply_theme
from ui.sections import (
    render_sidebar,
    render_tab_market_overview,
    render_tab_portfolio,
    render_tab_technical,
    render_tab_advice,
)
from analysis.recommender import get_client

apply_theme()

client = get_client()

render_sidebar()

# ── 标题栏 ──
st.title("📊 A股基金智能分析助手")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("为投资新手打造的 A 股基金分析工具 — 新闻 + 数据 + 技术指标 + AI 建议")
with col2:
    st.caption(f"⏱ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col3:
    has_ds = bool(os.getenv("DEEPSEEK_API_KEY"))
    has_ant = bool(os.getenv("ANTHROPIC_API_KEY"))
    api_ready = has_ds or has_ant
    if api_ready:
        label = "🟢 AI 分析就绪 (DeepSeek)" if has_ds else "🟢 AI 分析就绪 (Anthropic)"
    else:
        label = "🟠 AI 未配置（量化模式）"
    st.caption(label)

# ── 主内容区：4 个 Tab ──
tab1, tab2, tab3, tab4 = st.tabs(["市场概览", "我的持仓", "技术分析", "投资建议"])

with tab1:
    render_tab_market_overview(client)

with tab2:
    render_tab_portfolio()

with tab3:
    render_tab_technical()

with tab4:
    render_tab_advice(client)
