import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="A股基金智能分析助手",
    page_icon="📊",
    layout="wide",
)

from ui.sections import (
    render_block1_news,
    render_block2_screening,
    render_block3_portfolio,
    render_block4_technical,
    render_block5_advice,
)
from analysis.recommender import get_client

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

client = get_client()

render_block1_news(client)
st.divider()

render_block2_screening()
st.divider()

render_block3_portfolio()
st.divider()

render_block4_technical()
st.divider()

render_block5_advice(client)
