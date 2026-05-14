"""UI 区块渲染模块 — 市场要闻、基金筛选、持仓诊断、技术分析、投资建议"""

import streamlit as st
import pandas as pd
from analysis.portfolio import (
    PORTFOLIO_PATH,
    load_portfolio,
    add_holding,
    calc_portfolio_summary,
    diagnose_holding,
)
from data.fund_fetcher import (
    fetch_fund_rankings,
    fetch_fund_nav_history,
    get_fund_name,
)
from data.news_fetcher import fetch_market_news, format_news_for_ai
from data.technical import calc_ma, calc_macd, detect_cross, calc_max_drawdown
from analysis.screener import filter_and_rank
from analysis.recommender import generate_news_analysis, generate_investment_advice
from ui.charts import plot_nav_with_ma, plot_macd, plot_volume


# ---------------------------------------------------------------------------
# Block 1: 市场要闻与 AI 解读
# ---------------------------------------------------------------------------

def render_block1_news(client):
    """获取财经新闻，AI 解读市场情绪，展示新闻卡片"""
    st.header("📰 市场要闻解读")

    with st.spinner("正在获取最新财经新闻..."):
        try:
            news_df = fetch_market_news()
        except Exception as e:
            st.warning(f"新闻获取失败：{e}")
            return

    if news_df.empty:
        st.warning("暂无新闻数据，请检查网络连接")
        return

    news_text = format_news_for_ai(news_df)

    with st.spinner("AI 正在解读新闻..."):
        sentiment, news_items = generate_news_analysis(client, news_text)

    # 整体市场情绪徽章
    st.markdown(f"### 市场情绪：{sentiment}")

    # 新闻卡片，按情感颜色标记
    sentiment_color_map = {
        "利好": "#ef5350",
        "偏暖": "#ef5350",
        "利空": "#26a69a",
        "谨慎": "#26a69a",
        "中性": "#ff9800",
    }

    for item in news_items:
        s = item.get("sentiment", "中性")
        color = sentiment_color_map.get(s, "#9e9e9e")
        st.markdown(
            f"""<div style="
                border-left: 4px solid {color};
                padding: 10px 14px;
                margin: 8px 0;
                background: rgba(255,255,255,0.03);
                border-radius: 6px;
            ">
                <span style="color: {color}; font-weight: bold; font-size: 0.85em;">[{s}]</span>
                <strong> {item.get('title', '')}</strong><br>
                <small style="color: #888;">
                    {item.get('source', '')} · {item.get('time', '')} · 板块: {item.get('sector', '')}
                </small><br>
                <span style="color: #ccc;">{item.get('summary', '')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.session_state["market_sentiment"] = sentiment


# ---------------------------------------------------------------------------
# Block 2: 基金筛选与排名
# ---------------------------------------------------------------------------

def render_block2_screening():
    """筛选控件 + 基金排名 + 推荐列表"""
    st.header("🔍 基金筛选与排名")

    # ---- 筛选控件 ----
    col1, col2, col3 = st.columns(3)
    with col1:
        fund_type_options = [
            "股票型", "混合型", "债券型", "指数型", "QDII", "货币型", "ETF", "LOF",
        ]
        selected_types = st.multiselect(
            "基金类型（可多选）",
            options=fund_type_options,
            default=["股票型", "混合型"],
            help="选择要筛选的基金类型",
        )
    with col2:
        min_return = st.slider(
            "近1月最低收益 (%)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=0.5,
            help="筛选近1月收益不低于此值的基金",
        )
    with col3:
        min_scale = st.slider(
            "最低规模（亿元）",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=1.0,
            help="筛选规模不低于此值的基金",
        )

    # ---- 获取数据 ----
    with st.spinner("正在获取全市场基金排名..."):
        try:
            df = fetch_fund_rankings()
        except Exception as e:
            st.warning(f"基金数据获取失败：{e}")
            return

    if df.empty:
        st.warning("暂无基金排名数据，请稍后重试")
        return

    # 0 表示不限制
    min_return_val = min_return if min_return > 0 else None

    with st.spinner("正在筛选与评分..."):
        try:
            result = filter_and_rank(df, selected_types, min_return_val, min_scale)
        except Exception as e:
            st.warning(f"筛选过程出错：{e}")
            return

    if result.empty:
        st.warning("没有符合条件的基金，请放宽筛选条件")
        st.session_state["recommended_funds"] = []
        return

    # ---- 结果展示 ----
    st.subheader(f"筛选结果（Top {len(result)}）")

    display_cols = [
        "基金代码", "基金简称", "基金类型", "单位净值",
        "日增长率", "近1月", "近3月", "基金规模", "score",
    ]
    available_cols = [c for c in display_cols if c in result.columns]
    display_df = result[available_cols].reset_index(drop=True)

    # 第 1 名绿色高亮，第 2 名黄色高亮
    def row_highlight(row):
        if row.name == 0:
            return ["background-color: rgba(76, 175, 80, 0.12)" for _ in row]
        if row.name == 1:
            return ["background-color: rgba(255, 235, 59, 0.10)" for _ in row]
        return ["" for _ in row]

    styled_df = display_df.style.apply(row_highlight, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

    # ---- 推荐基金 ----
    top5 = result.head(5)
    st.session_state["recommended_funds"] = top5.to_dict(orient="records")

    st.subheader("⭐ 推荐关注")
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, (_, fund) in enumerate(top5.iterrows()):
        st.markdown(
            f"{medals[i]} **{fund.get('基金简称', '')}** "
            f"({fund.get('基金代码', '')})"
            f" — 评分: {fund.get('score', 0):.1f}"
            f" | 近1月: {fund.get('近1月', '')}"
            f" | 近3月: {fund.get('近3月', '')}"
            f" | 规模: {fund.get('基金规模', '')}"
        )


# ---------------------------------------------------------------------------
# Block 3: 我的持仓与诊断
# ---------------------------------------------------------------------------

def render_block3_portfolio():
    """添加/展示持仓，计算汇总，四维度诊断每只基金"""
    st.header("📊 我的持仓")

    # ---- 添加持仓表单 ----
    with st.form("add_holding_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            code = st.text_input("基金代码", placeholder="例如: 000001", key="input_code")
        with c2:
            cost_nav = st.number_input(
                "成本净值", min_value=0.0001, step=0.0001, format="%.4f",
                key="input_cost_nav",
            )
        with c3:
            amount = st.number_input(
                "投入金额", min_value=0.01, step=100.0, format="%.2f",
                key="input_amount",
            )
        with c4:
            st.write("")  # 占位对齐
            st.write("")
            submitted = st.form_submit_button("➕ 添加", use_container_width=True)

        if submitted:
            if code and cost_nav > 0 and amount > 0:
                try:
                    name = get_fund_name(code)
                    add_holding(PORTFOLIO_PATH, code, name, cost_nav, amount)
                    st.success(f"已添加：{name} ({code})")
                except Exception as e:
                    st.warning(f"添加失败：{e}")
            else:
                st.warning("请填写完整的基金信息（代码、成本净值、金额）")

    # ---- 加载持仓 ----
    portfolio = load_portfolio(PORTFOLIO_PATH)
    holdings = portfolio.get("holdings", [])

    if not holdings:
        st.info("暂无持仓记录，请先添加基金")
        st.session_state["diagnosis_results"] = []
        return

    # ---- 持仓表格 ----
    st.subheader(f"当前持仓（{len(holdings)} 只）")
    holdings_data = []
    for h in holdings:
        holdings_data.append({
            "基金代码": h["code"],
            "基金名称": h.get("name", h["code"]),
            "成本净值": h["cost_nav"],
            "投入金额": h["amount"],
            "添加日期": h.get("added_date", ""),
        })
    st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True)

    # ---- 获取最新净值 ----
    current_navs: dict[str, float] = {}
    with st.spinner("正在获取最新净值..."):
        try:
            rankings_df = fetch_fund_rankings()
            if not rankings_df.empty:
                for h in holdings:
                    match = rankings_df[rankings_df["基金代码"] == h["code"]]
                    if not match.empty:
                        nav_val = match.iloc[0].get("单位净值", 0)
                        try:
                            nav_f = float(nav_val)
                            if nav_f > 0:
                                current_navs[h["code"]] = nav_f
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            st.warning(f"获取最新净值失败：{e}")

    # ---- 组合汇总 ----
    summary = calc_portfolio_summary(PORTFOLIO_PATH, current_navs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总投入", f"¥{summary['total_invested']:,.2f}")
    c2.metric("当前市值", f"¥{summary['total_value']:,.2f}")
    c3.metric(
        "总盈亏",
        f"¥{summary['total_pnl']:,.2f}",
        delta=f"{summary['total_pnl_pct']:.2f}%",
    )
    c4.metric("持仓数量", f"{summary['holding_count']} 只")

    # ---- 逐只诊断 ----
    st.subheader("🔬 持仓诊断")
    diagnosis_results = []

    with st.spinner("正在逐只诊断（获取历史净值、计算技术指标）..."):
        for h in holdings:
            code = h["code"]
            cost_nav = h["cost_nav"]
            name = h.get("name", code)

            try:
                nav_df = fetch_fund_nav_history(code, days=90)

                if nav_df.empty or len(nav_df) < 10:
                    st.warning(f"{name} ({code})：历史数据不足，跳过诊断")
                    diagnosis_results.append({
                        "code": code, "name": name, "cost_nav": cost_nav,
                        "error": True, "message": "历史数据不足",
                    })
                    continue

                nav_df = calc_ma(nav_df)
                nav_df = calc_macd(nav_df)

                nav_current = float(nav_df["单位净值"].iloc[-1])

                ma10_series = nav_df["MA10"].dropna()
                ma20_series = nav_df["MA20"].dropna()
                ma10 = float(ma10_series.iloc[-1]) if len(ma10_series) > 0 else nav_current
                ma20 = float(ma20_series.iloc[-1]) if len(ma20_series) > 0 else nav_current

                # 近 1 月收益（约 22 个交易日）
                if len(nav_df) >= 22:
                    nav_1m_ago = float(nav_df["单位净值"].iloc[-22])
                    return_1m = (nav_current - nav_1m_ago) / nav_1m_ago if nav_1m_ago > 0 else 0.0
                else:
                    return_1m = 0.0

                benchmark_return = 0.0  # 后续可接指数基准

                # 近 1 月最大回撤
                recent_1m = nav_df["单位净值"].iloc[-22:] if len(nav_df) >= 22 else nav_df["单位净值"]
                max_dd = calc_max_drawdown(recent_1m)

                diag = diagnose_holding(
                    nav_current=nav_current,
                    nav_cost=cost_nav,
                    return_1m=return_1m,
                    benchmark_return=benchmark_return,
                    ma10=ma10,
                    ma20=ma20,
                    max_drawdown_1m=max_dd,
                )

                diagnosis_results.append({
                    "code": code,
                    "name": name,
                    "cost_nav": cost_nav,
                    "nav_current": round(nav_current, 4),
                    "pnl_pct": diag["pnl_pct"],
                    "score": diag["score"],
                    "label": diag["label"],
                    "suggestion": diag["suggestion"],
                    "return_1m": round(return_1m * 100, 2),
                    "max_drawdown": round(max_dd * 100, 2),
                    "error": False,
                })

            except Exception as e:
                st.warning(f"{name} ({code})：诊断失败 - {e}")
                diagnosis_results.append({
                    "code": code,
                    "name": name,
                    "cost_nav": cost_nav,
                    "error": True,
                    "message": str(e),
                })

    # ---- 诊断结果表格 ----
    valid_results = [r for r in diagnosis_results if not r.get("error")]
    if valid_results:
        diag_df = pd.DataFrame(valid_results)
        display_diag = diag_df[[
            "code", "name", "nav_current", "pnl_pct",
            "return_1m", "max_drawdown", "score", "label", "suggestion",
        ]]
        display_diag.columns = [
            "代码", "名称", "最新净值", "盈亏%", "近1月%", "最大回撤%", "评分", "建议标签", "操作建议"
        ]

        def color_label(val: str) -> str:
            if "持有加仓" in str(val):
                return "color: #4caf50; font-weight: bold"
            if "暂持观望" in str(val):
                return "color: #ff9800; font-weight: bold"
            if "减仓" in str(val):
                return "color: #ff5722; font-weight: bold"
            if "清仓" in str(val):
                return "color: #f44336; font-weight: bold"
            return ""

        styled_diag = display_diag.style.map(color_label, subset=["建议标签"])
        st.dataframe(styled_diag, use_container_width=True, hide_index=True)

        # 诊断说明
        with st.expander("评分说明"):
            st.markdown("""
            | 分数 | 建议 | 含义 |
            |------|------|------|
            | >= 75 | 🟢 持有加仓 | 趋势向好，可继续持有或适当加仓 |
            | 55-74 | 🟡 暂持观望 | 信号不明确，建议等待 |
            | 35-54 | 🟠 减仓 | 风险偏高，建议降低仓位 |
            | < 35 | 🔴 清仓 | 趋势恶化，建议止损离场 |
            """)

    st.session_state["diagnosis_results"] = diagnosis_results
    st.session_state["current_navs"] = current_navs


# ---------------------------------------------------------------------------
# Block 4: 技术分析图表
# ---------------------------------------------------------------------------

def render_block4_technical():
    """为持仓和推荐基金绘制净值/MA、MACD、成交量图表，检测交叉信号"""
    st.header("📈 技术分析")

    diagnosis_results = st.session_state.get("diagnosis_results", [])
    recommended_funds = st.session_state.get("recommended_funds", [])

    # 收集需要绘图的基金（去重：优先持仓，补充推荐）
    funds_to_analyze: list[dict] = []
    existing_codes: set[str] = set()

    for r in diagnosis_results:
        if not r.get("error"):
            funds_to_analyze.append({
                "code": r["code"],
                "name": r["name"],
                "cost_nav": r.get("cost_nav"),
            })
            existing_codes.add(r["code"])

    for fund in recommended_funds[:3]:
        code = fund.get("基金代码", "")
        if code and code not in existing_codes:
            funds_to_analyze.append({
                "code": code,
                "name": fund.get("基金简称", code),
                "cost_nav": None,
            })
            existing_codes.add(code)

    if not funds_to_analyze:
        st.info("暂无数据可分析，请先添加持仓或进行基金筛选")
        return

    all_signals: list[str] = []

    for fund in funds_to_analyze:
        code = fund["code"]
        name = fund["name"]
        cost_nav = fund.get("cost_nav")

        with st.spinner(f"正在分析 {name} ({code})..."):
            try:
                nav_df = fetch_fund_nav_history(code, days=90)

                if nav_df.empty or len(nav_df) < 20:
                    st.warning(f"{name} ({code})：历史数据不足（需 ≥20 个交易日），跳过")
                    continue

                nav_df = calc_ma(nav_df)
                nav_df = calc_macd(nav_df)

                # 净值 + 均线
                fig_nav = plot_nav_with_ma(nav_df, name, cost_line=cost_nav)
                st.plotly_chart(fig_nav, use_container_width=True)

                # MACD
                fig_macd = plot_macd(nav_df, name)
                st.plotly_chart(fig_macd, use_container_width=True)

                # 成交量
                fig_vol = plot_volume(nav_df, name)
                st.plotly_chart(fig_vol, use_container_width=True)

                # ---- 交叉信号检测 ----
                crosses_ma10 = detect_cross(nav_df["单位净值"], nav_df["MA10"])
                crosses_ma20 = detect_cross(nav_df["单位净值"], nav_df["MA20"])

                if "DIF" in nav_df.columns and "DEA" in nav_df.columns:
                    crosses_macd = detect_cross(nav_df["DIF"], nav_df["DEA"])
                else:
                    crosses_macd = []

                # 最近 5 个交易日的信号
                recent_date = nav_df["净值日期"].iloc[-1]
                if isinstance(recent_date, pd.Timestamp):
                    five_days_ago = recent_date - pd.Timedelta(days=7)  # 自然日留余量
                else:
                    five_days_ago = recent_date

                recent_signals: list[str] = []
                for cross_list, label in [
                    (crosses_ma10, "MA10"),
                    (crosses_ma20, "MA20"),
                    (crosses_macd, "MACD"),
                ]:
                    for c in cross_list:
                        idx = c["index"]
                        if isinstance(idx, pd.Timestamp) and idx >= five_days_ago:
                            ctype = "金叉🟢" if c["type"] == "golden_cross" else "死叉🔴"
                            recent_signals.append(
                                f"{idx.strftime('%m-%d')} {label}{ctype}"
                            )

                if recent_signals:
                    st.caption(f"近期信号: {' | '.join(recent_signals)}")
                    all_signals.append(f"{name}: {'; '.join(recent_signals)}")
                else:
                    st.caption("近期无明确交叉信号")

                st.markdown("---")

            except Exception as e:
                st.warning(f"{name} ({code})：技术分析失败 - {e}")

    st.session_state["technical_signals"] = (
        "\n".join(all_signals) if all_signals else "无明确技术信号"
    )


# ---------------------------------------------------------------------------
# Block 5: 综合投资建议
# ---------------------------------------------------------------------------

def render_block5_advice(client):
    """汇总所有分析数据，调用 AI 生成投资建议"""
    st.header("💡 投资建议")

    market_sentiment = st.session_state.get("market_sentiment", "🟡 中性")
    diagnosis_results = st.session_state.get("diagnosis_results", [])
    recommended_funds = st.session_state.get("recommended_funds", [])
    technical_signals = st.session_state.get("technical_signals", "无明确技术信号")

    # 格式化推荐基金信息
    top_funds_lines: list[str] = []
    for fund in recommended_funds[:3]:
        top_funds_lines.append(
            f"- {fund.get('基金简称', '')} ({fund.get('基金代码', '')}) "
            f"评分: {fund.get('score', 0)} | "
            f"近1月: {fund.get('近1月', '')} | "
            f"近3月: {fund.get('近3月', '')}"
        )
    top_funds_info = "\n".join(top_funds_lines) if top_funds_lines else "暂无推荐基金"

    # 格式化持仓诊断信息
    diag_lines: list[str] = []
    for r in diagnosis_results:
        if r.get("error"):
            diag_lines.append(
                f"- {r.get('name', r.get('code', ''))}: "
                f"诊断失败 - {r.get('message', '')}"
            )
        else:
            diag_lines.append(
                f"- {r['name']} ({r['code']}): "
                f"评分 {r['score']} | 盈亏 {r['pnl_pct']}% | "
                f"{r['label']} | {r['suggestion']}"
            )
    portfolio_diagnosis = "\n".join(diag_lines) if diag_lines else "暂无持仓"

    with st.spinner("AI 正在生成综合投资建议..."):
        advice = generate_investment_advice(
            client,
            market_sentiment=market_sentiment,
            top_funds_info=top_funds_info,
            portfolio_diagnosis=portfolio_diagnosis,
            technical_signals=technical_signals,
        )

    st.markdown(advice)

    st.markdown("---")
    st.caption(
        "⚠️ 免责声明：本工具基于历史数据和技术指标生成分析，不构成投资建议。"
        "历史收益不代表未来表现。基金投资有风险，入市需谨慎。"
        "请结合自身风险承受能力做出投资决策。"
    )
