"""UI 区块渲染模块 — 侧边栏 + 4 个 Tab（市场概览、持仓、技术分析、投资建议）"""

import streamlit as st
import pandas as pd
from datetime import datetime
from analysis.portfolio import (
    PORTFOLIO_PATH,
    load_portfolio,
    add_holding,
    remove_holding,
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
from analysis.prediction_store import (
    smooth_and_store,
    get_smoothed_prediction,
    get_long_term_suggestion,
    ACTION_LABELS,
)
from ui.charts import plot_nav_with_ma, plot_macd, plot_volume
from analysis.watchlist import load_watchlist, pin_fund, unpin_fund, is_pinned, get_pinned_funds
from analysis.screening_prefs import load_prefs, save_prefs
from ui.theme import COLORS, SENTIMENT_COLORS, LABEL_COLORS


# ════════════════════════════════════════════════════════════
# 侧边栏
# ════════════════════════════════════════════════════════════

def render_sidebar():
    """侧边栏：持仓迷你概览 + 快捷操作"""
    with st.sidebar:
        st.markdown("#### 📊 我的持仓概览")

        try:
            from analysis.portfolio import PORTFOLIO_PATH, load_portfolio
            portfolio = load_portfolio(PORTFOLIO_PATH)
            holdings = portfolio.get("holdings", [])

            if holdings:
                st.metric("持仓数量", f"{len(holdings)} 只")

                # 尝试获取当前市值
                try:
                    rankings_df = fetch_fund_rankings()
                    total_value = 0.0
                    total_cost = 0.0
                    for h in holdings:
                        code = h["code"]
                        nav = h.get("cost_nav", 0)
                        amount = h.get("amount", 0)
                        match = rankings_df[rankings_df["基金代码"] == code]
                        if not match.empty:
                            current_nav = float(match.iloc[0].get("单位净值", nav))
                            total_value += (amount / nav) * current_nav
                        else:
                            total_value += amount
                        total_cost += amount

                    total_pnl = total_value - total_cost
                    pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
                    st.metric(
                        "总市值", f"¥{total_value:,.2f}",
                        delta=f"{pnl_pct:+.2f}%",
                    )
                except Exception:
                    st.caption("市值计算暂不可用")
            else:
                st.caption("暂无持仓记录")
        except Exception:
            st.caption("无法加载持仓")

        # ── 锁定追踪 ──
        st.markdown("---")
        st.markdown("#### 📌 锁定追踪")
        pinned = get_pinned_funds()
        if pinned:
            for f in pinned:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"{f['name']} ({f['code']})")
                with col2:
                    if st.button("🔓", key=f"sidebar_unpin_{f['code']}", help=f"解锁 {f['name']}"):
                        unpin_fund(f["code"])
                        st.rerun()
        else:
            st.caption("暂无锁定基金，在筛选结果中点击 📌 即可追踪")

        # 市场情绪（仅在有持仓或锁定基金时显示，避免空态误导）
        has_holdings = False
        try:
            portfolio = load_portfolio(PORTFOLIO_PATH)
            has_holdings = len(portfolio.get("holdings", [])) > 0
        except Exception:
            pass
        if has_holdings or len(pinned) > 0:
            sentiment = st.session_state.get("market_sentiment", "")
            if sentiment:
                st.markdown(f"##### 市场情绪：{sentiment}")

        st.markdown("---")

        # 强制刷新
        if st.button("🔄 刷新全部数据", use_container_width=True):
            from data.fund_fetcher import fetch_fund_rankings as fr, fetch_fund_nav_history as fn
            fr.invalidate()
            fn.invalidate()
            st.rerun()

        st.markdown("---")
        st.caption("A股基金智能分析助手 v2.0")


# ════════════════════════════════════════════════════════════
# Tab 1: 市场概览（新闻 + 筛选）
# ════════════════════════════════════════════════════════════

def render_tab_market_overview(client):
    """市场要闻 + AI 解读 + 基金筛选与排名"""

    # ── Part A: 市场要闻 ──
    st.header("📰 市场要闻解读")

    with st.spinner("正在获取最新财经新闻..."):
        try:
            news_df = fetch_market_news()
        except Exception as e:
            st.warning(f"新闻获取失败：{e}")
            return

    if news_df.empty:
        st.warning("暂无新闻数据，请检查网络连接")
    else:
        news_text = format_news_for_ai(news_df)

        with st.spinner("AI 正在解读新闻..."):
            sentiment, news_items = generate_news_analysis(client, news_text)

        st.markdown(f"### 市场情绪：{sentiment}")
        st.session_state["market_sentiment"] = sentiment

        for item in news_items:
            s = item.get("sentiment", "中性")
            color = SENTIMENT_COLORS.get(s, COLORS["muted"])
            st.markdown(
                f"""<div style="
                    border-left: 4px solid {color};
                    padding: 10px 14px;
                    margin: 8px 0;
                    background: {COLORS["bg_card"]};
                    border-radius: 6px;
                ">
                    <span style="color: {color}; font-weight: bold; font-size: 0.85em;">[{s}]</span>
                    <strong> {item.get('title', '')}</strong><br>
                    <small style="color: {COLORS['text_secondary']};">
                        {item.get('source', '')} · {item.get('time', '')} · 板块: {item.get('sector', '')}
                    </small><br>
                    <span style="color: {COLORS['text_primary']};">{item.get('summary', '')}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Part B: 基金筛选与排名 ──
    st.header("🔍 基金筛选与排名")

    # ── 加载持久化筛选偏好 ──
    prefs = load_prefs()
    if "screening_types" not in st.session_state:
        st.session_state.screening_types = prefs["fund_types"]
    if "screening_min_return" not in st.session_state:
        st.session_state.screening_min_return = prefs["min_return"]
    if "screening_min_scale" not in st.session_state:
        st.session_state.screening_min_scale = prefs["min_scale"]

    col1, col2, col3 = st.columns(3)
    with col1:
        fund_type_options = [
            "股票型", "混合型", "债券型", "指数型", "QDII", "货币型", "ETF", "LOF",
        ]
        selected_types = st.multiselect(
            "基金类型（可多选）",
            options=fund_type_options,
            key="screening_types",
            help="选择要筛选的基金类型",
        )
    with col2:
        min_return = st.slider(
            "近1月最低收益 (%)",
            min_value=0.0,
            max_value=30.0,
            step=0.5,
            key="screening_min_return",
            help="筛选近1月收益不低于此值的基金",
        )
    with col3:
        min_scale = st.slider(
            "最低规模（亿元）",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key="screening_min_scale",
            help="筛选规模不低于此值的基金",
        )

        enable_prediction = st.checkbox(
            "启用盈利预测（耗时约15-30秒）",
            value=False,
            help="基于历史模式匹配预测 1/2/3 月盈利概率",
        )

    # 持久化当前筛选值
    save_prefs({
        "fund_types": st.session_state.screening_types,
        "min_return": st.session_state.screening_min_return,
        "min_scale": st.session_state.screening_min_scale,
    })

    with st.spinner("正在获取全市场基金排名..."):
        try:
            df = fetch_fund_rankings()
        except Exception as e:
            st.warning(f"基金数据获取失败：{e}")
            return

    if df.empty:
        st.warning("暂无基金排名数据，请稍后重试")
        return

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

    # 结果展示
    st.subheader(f"筛选结果（Top {len(result)}）")

    display_cols = [
        "基金代码", "基金简称", "基金类型", "单位净值",
        "日增长率", "近1月", "近3月", "基金规模", "score",
    ]
    available_cols = [c for c in display_cols if c in result.columns]
    display_df = result[available_cols].reset_index(drop=True)

    def row_highlight(row):
        if row.name == 0:
            return [f"background-color: rgba(0, 212, 170, 0.08)" for _ in row]
        if row.name == 1:
            return [f"background-color: rgba(255, 217, 61, 0.06)" for _ in row]
        return ["" for _ in row]

    styled_df = display_df.style.apply(row_highlight, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

    # 推荐基金（市场参考）
    top5 = result.head(5)
    st.session_state["recommended_funds"] = top5.to_dict(orient="records")

    st.subheader("📋 市场参考（今日快照）")
    st.caption("提示：以下为今日市场扫描结果，推荐关注名单会随市场变化。持仓操作请以「我的持仓」Tab 中的长线信号为准。")

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, (_, fund) in enumerate(top5.iterrows()):
        code = str(fund.get('基金代码', ''))
        name = str(fund.get('基金简称', ''))
        pinned = is_pinned(code)
        c1, c2 = st.columns([20, 1])
        with c1:
            st.markdown(
                f"{medals[i]} **{name}** "
                f"({code})"
                f" — 评分: {fund.get('score', 0):.1f}"
                f" | 近1月: {fund.get('近1月', '')}"
                f" | 近3月: {fund.get('近3月', '')}"
                f" | 规模: {fund.get('基金规模', '')}"
            )
        with c2:
            if pinned:
                if st.button("🔓", key=f"unpin_top_{code}", help=f"解锁 {name}"):
                    unpin_fund(code)
                    st.rerun()
            else:
                if st.button("📌", key=f"pin_top_{code}", help=f"锁定 {name} 以便连续追踪"):
                    pin_fund(code, name, source="screening")
                    st.rerun()

    # 盈利预测
    if enable_prediction:
        predictions = []
        pred_dict_for_store = {}

        with st.spinner("正在进行盈利预测（模式匹配 + 统计推断）..."):
            try:
                from analysis.predictor import predict_fund_return
                from data.fund_fetcher import fetch_benchmark_data

                benchmark_nav = fetch_benchmark_data(days=365)
                for _, fund in result.iterrows():
                    code = fund["基金代码"]
                    name = fund.get("基金简称", code)
                    try:
                        nav_history = fetch_fund_nav_history(str(code), days=365)
                        if nav_history.empty or len(nav_history) < 60:
                            continue
                        nav_series = nav_history.set_index("净值日期")["单位净值"]
                        pred = predict_fund_return(nav_series, benchmark_nav, code=str(code), name=str(name))
                        predictions.append(pred)

                        # 构建用于平滑存储的 dict
                        if pred.pred_1m:
                            pred_dict_for_store[str(code)] = {
                                "name": str(name),
                                "win_prob_1m": pred.pred_1m.win_probability,
                                "win_prob_2m": pred.pred_2m.win_probability if pred.pred_2m else 0.5,
                                "win_prob_3m": pred.pred_3m.win_probability if pred.pred_3m else 0.5,
                                "median_return_1m": pred.pred_1m.median_return,
                                "median_return_2m": pred.pred_2m.median_return if pred.pred_2m else 0.0,
                                "median_return_3m": pred.pred_3m.median_return if pred.pred_3m else 0.0,
                                "confidence": pred.confidence,
                            }
                    except Exception:
                        continue

                # 平滑并持久化预測
                if pred_dict_for_store:
                    smooth_and_store(pred_dict_for_store)

                if predictions:
                    st.subheader("盈利预测（含历史平滑）")
                    st.caption("📌 点击锁定可跨天追踪 | 预测已平滑避免单日波动 | 趋势信号需连续3天确认后变更")

                    st.session_state["_now_iso"] = datetime.now().isoformat()

                    # ── 自定义表格：锁定按钮在最左列 ──
                    col_ratios = [0.35, 0.7, 1.8, 1.1, 1.1, 1.1, 1.1]
                    hdr = st.columns(col_ratios)
                    hdr[0].caption("")
                    hdr[1].caption("代码")
                    hdr[2].caption("名称")
                    hdr[3].caption("近1月")
                    hdr[4].caption("近2月")
                    hdr[5].caption("近3月")
                    hdr[6].caption("长线信号")

                    for p in predictions:
                        code = str(p.code)
                        name = str(p.name)
                        pinned = is_pinned(code)
                        smoothed = get_smoothed_prediction(code)

                        row = st.columns(col_ratios)

                        # Col 0: 锁定按钮
                        with row[0]:
                            if pinned:
                                if st.button("🔓", key=f"pred_unpin_{code}", help=f"解锁 {name}"):
                                    unpin_fund(code)
                                    st.rerun()
                            else:
                                if st.button("📌", key=f"pred_pin_{code}", help=f"锁定 {name} 以便连续追踪"):
                                    pin_fund(code, name, source="prediction")
                                    st.rerun()

                        # Col 1: 代码
                        with row[1]:
                            st.caption(code)

                        # Col 2: 名称
                        with row[2]:
                            st.caption(name)

                        # Cols 3-5: 预测数据
                        for ci, (period_key, label) in enumerate([("pred_1m", "近1月"), ("pred_2m", "近2月"), ("pred_3m", "近3月")]):
                            pp = getattr(p, period_key)
                            with row[3 + ci]:
                                if pp:
                                    prob = pp.win_probability
                                    if smoothed:
                                        sp_key = f"win_prob_{'1m' if '1月' in label else '2m' if '2月' in label else '3m'}"
                                        smoothed_prob = smoothed.get(sp_key, prob)
                                        diff = prob - smoothed_prob
                                        trend = " ↑" if abs(diff) > 0.02 else ""
                                        val_text = f"{prob:.0%}{trend}"
                                    else:
                                        arrow = " ↑" if pp.median_return > 0 else " ↓"
                                        val_text = f"{pp.win_probability:.0%}{arrow}"

                                    if prob >= 0.60:
                                        color = COLORS["primary"]
                                    elif prob >= 0.40:
                                        color = COLORS["warning"]
                                    else:
                                        color = COLORS["muted"]
                                    st.markdown(
                                        f"<span style='color:{color};font-weight:bold;font-size:0.9em'>{val_text}</span>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.caption("N/A")

                        # Col 6: 长线信号
                        with row[6]:
                            if smoothed:
                                action = smoothed.get("recommended_action", "hold_wait")
                                bullish = smoothed.get("consecutive_bullish_days", 0)
                                bearish = smoothed.get("consecutive_bearish_days", 0)
                                locked = smoothed.get("action_locked_until", "")
                                if locked and st.session_state.get("_now_iso", "") < locked:
                                    signal_text = f"🔒 {ACTION_LABELS.get(action, action)}"
                                    sig_color = COLORS["primary"]
                                elif bullish >= 2:
                                    signal_text = f"🟢 {bullish}天看涨"
                                    sig_color = COLORS["primary"]
                                elif bearish >= 2:
                                    signal_text = f"🔴 {bearish}天看跌"
                                    sig_color = COLORS["danger"]
                                else:
                                    signal_text = "🟡 待确认"
                                    sig_color = COLORS["warning"]
                                st.markdown(
                                    f"<span style='color:{sig_color};font-size:0.85em'>{signal_text}</span>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.caption("—")

                    st.session_state["predictions"] = predictions
            except Exception as e:
                st.warning(f"预测失败：{e}")


# ════════════════════════════════════════════════════════════
# Tab 2: 我的持仓
# ════════════════════════════════════════════════════════════

def render_tab_portfolio():
    """添加/展示持仓，计算汇总，四维度诊断 + 长线预测信号"""
    st.header("📊 我的持仓")

    # ── 添加持仓表单 ──
    with st.form("add_holding_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            code = st.text_input("基金代码", placeholder="例如: 000001", key="input_code")
        with c2:
            cost_nav = st.number_input(
                "成本净值", min_value=0.0001, step=0.0001, format="%.4f",
                help="您买入该基金时的单位净值，用于计算持仓盈亏",
                key="input_cost_nav",
            )
        with c3:
            amount = st.number_input(
                "投入金额", min_value=0.01, step=100.0, format="%.2f",
                key="input_amount",
            )
        with c4:
            st.write("")
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

    # ── 加载持仓 ──
    portfolio = load_portfolio(PORTFOLIO_PATH)
    holdings = portfolio.get("holdings", [])

    if not holdings:
        st.info("暂无持仓记录，请先添加基金")
        st.session_state["diagnosis_results"] = []
        st.session_state["current_navs"] = {}
        return

    # ── 持仓表格（含删除按钮） ──
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

    # 删除按钮（每只基金一个）
    cols = st.columns(min(len(holdings), 4))
    for i, h in enumerate(holdings):
        with cols[i % 4]:
            if st.button(f"🗑 删除 {h['code']}", key=f"remove_{h['code']}"):
                remove_holding(PORTFOLIO_PATH, h["code"])
                st.rerun()

    # ── 获取最新净值 ──
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

    # ── 组合汇总 ──
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

    # ── 逐只诊断 + 长线预测 ──
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

                if len(nav_df) >= 22:
                    nav_1m_ago = float(nav_df["单位净值"].iloc[-22])
                    return_1m = (nav_current - nav_1m_ago) / nav_1m_ago if nav_1m_ago > 0 else 0.0
                else:
                    return_1m = 0.0

                benchmark_return = 0.0

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

                # 获取长线预测信号
                long_term_label, long_term_color, long_term_note = get_long_term_suggestion(
                    diag["score"], code
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
                    "long_term_label": long_term_label,
                    "long_term_note": long_term_note,
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

    # ── 诊断结果表格 ──
    valid_results = [r for r in diagnosis_results if not r.get("error")]
    if valid_results:
        diag_df = pd.DataFrame(valid_results)
        display_diag = diag_df[[
            "code", "name", "nav_current", "pnl_pct",
            "return_1m", "max_drawdown", "score", "label",
            "long_term_label", "suggestion",
        ]]
        display_diag.columns = [
            "代码", "名称", "最新净值", "盈亏%", "近1月%", "最大回撤%",
            "评分", "短期建议", "长线信号", "操作建议",
        ]

        def color_label(val: str) -> str:
            if "持有加仓" in str(val):
                return f"color: {COLORS['primary']}; font-weight: bold"
            if "暂持观望" in str(val):
                return f"color: {COLORS['warning']}; font-weight: bold"
            if "减仓" in str(val):
                return f"color: {COLORS['danger']}; font-weight: bold"
            if "清仓" in str(val):
                return f"color: {COLORS['danger']}; font-weight: bold"
            return ""

        styled_diag = display_diag.style.map(color_label, subset=["短期建议", "长线信号"])
        st.dataframe(styled_diag, use_container_width=True, hide_index=True)

        # 长线信号说明
        for r in valid_results:
            if r.get("long_term_note"):
                st.caption(f"📌 {r['name']}：{r['long_term_note']}")

        with st.expander("评分说明"):
            st.markdown(f"""
            | 分数 | 建议 | 含义 |
            |------|------|------|
            | >= 75 | 🟢 持有加仓 | 趋势向好，可继续持有或适当加仓 |
            | 55-74 | 🟡 暂持观望 | 信号不明确，建议等待 |
            | 35-54 | 🟠 减仓 | 风险偏高，建议降低仓位 |
            | < 35 | 🔴 清仓 | 趋势恶化，建议止损离场 |

            **长线信号说明：** 基于盈利预测的平滑结果（30%新数据 + 70%历史加权）。
            看涨/看跌信号需连续3天确认后才会变更建议，变更后锁定7天。这样避免了短期市场波动导致的频繁操作。
            """)

    st.session_state["diagnosis_results"] = diagnosis_results
    st.session_state["current_navs"] = current_navs


# ════════════════════════════════════════════════════════════
# Tab 3: 技术分析
# ════════════════════════════════════════════════════════════

def render_tab_technical():
    """为持仓和推荐基金绘制技术图表，每只基金用折叠面板包裹"""
    st.header("📈 技术分析")

    diagnosis_results = st.session_state.get("diagnosis_results", [])
    recommended_funds = st.session_state.get("recommended_funds", [])

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

    # 锁定追踪基金
    for f in get_pinned_funds():
        code = f["code"]
        if code not in existing_codes:
            funds_to_analyze.append({
                "code": code,
                "name": f["name"],
                "cost_nav": None,
                "pinned": True,
            })
            existing_codes.add(code)

    if not funds_to_analyze:
        st.info("暂无数据可分析，请先添加持仓或进行基金筛选")
        return

    predictions = st.session_state.get("predictions", [])
    pred_map = {str(p.code): p for p in predictions}

    all_signals: list[str] = []

    for i, fund in enumerate(funds_to_analyze):
        code = fund["code"]
        name = fund["name"]
        cost_nav = fund.get("cost_nav")

        # 构建折叠面板标题
        nav_str = ""
        for r in diagnosis_results:
            if r.get("code") == code and not r.get("error"):
                nav_str = f"净值: ¥{r.get('nav_current', '—')}"
                break

        cost_str = f"成本: ¥{cost_nav:.4f}" if cost_nav else ""
        pinned_mark = "📌 " if (fund.get("pinned") or is_pinned(code)) else ""
        header = f"{pinned_mark}📈 {name} ({code}) — {nav_str} | {cost_str}" if nav_str or cost_str else f"{pinned_mark}📈 {name} ({code})"

        with st.expander(header, expanded=(i == 0)):
            with st.spinner(f"正在分析 {name} ({code})..."):
                try:
                    nav_df = fetch_fund_nav_history(code, days=90)

                    if nav_df.empty or len(nav_df) < 20:
                        st.warning(f"{name} ({code})：历史数据不足（需 ≥20 个交易日），跳过")
                        continue

                    nav_df = calc_ma(nav_df)
                    nav_df = calc_macd(nav_df)

                    fig_nav = plot_nav_with_ma(nav_df, name, cost_line=cost_nav)
                    st.plotly_chart(fig_nav, use_container_width=True)

                    fig_macd = plot_macd(nav_df, name)
                    st.plotly_chart(fig_macd, use_container_width=True)

                    fig_vol = plot_volume(nav_df, name)
                    st.plotly_chart(fig_vol, use_container_width=True)

                    # 交叉信号检测
                    crosses_ma10 = detect_cross(nav_df["单位净值"], nav_df["MA10"])
                    crosses_ma20 = detect_cross(nav_df["单位净值"], nav_df["MA20"])

                    if "DIF" in nav_df.columns and "DEA" in nav_df.columns:
                        crosses_macd = detect_cross(nav_df["DIF"], nav_df["DEA"])
                    else:
                        crosses_macd = []

                    recent_date = nav_df["净值日期"].iloc[-1]
                    if isinstance(recent_date, pd.Timestamp):
                        five_days_ago = recent_date - pd.Timedelta(days=7)
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

                    # 盈利预测卡片
                    pred = pred_map.get(code)
                    if pred and pred.pred_1m:
                        from ui.charts import plot_return_distribution

                        st.markdown(f"##### 📈 盈利预测 — 置信度: {pred.confidence}")
                        c1, c2, c3, c4 = st.columns(4)
                        pp_list = [("近1月", pred.pred_1m), ("近2月", pred.pred_2m), ("近3月", pred.pred_3m)]
                        for ci, (label, pp) in enumerate(pp_list):
                            with [c1, c2, c3][ci]:
                                color = COLORS["primary"] if pp.win_probability >= 0.6 else (
                                    COLORS["warning"] if pp.win_probability >= 0.4 else COLORS["muted"]
                                )
                                st.metric(
                                    label=label,
                                    value=f"{pp.win_probability:.0%}",
                                    delta=f"{pp.median_return:+.1%}",
                                )
                        with c4:
                            st.metric("平均相似度", f"{pred.avg_similarity:.2f}")

                        st.caption(
                            f"预期区间: {pred.pred_1m.p25_return:+.1%} ~ {pred.pred_1m.p75_return:+.1%} | "
                            f"基于 {pred.match_count} 个最相似历史情景"
                        )

                        fig_dist = plot_return_distribution(pred.pred_1m, pred.pred_2m, pred.pred_3m, name)
                        st.plotly_chart(fig_dist, use_container_width=True)

                        if pred.confidence == "低":
                            st.warning("相似度较低，预测结果仅供参考")

                except Exception as e:
                    st.warning(f"{name} ({code})：技术分析失败 - {e}")

    st.session_state["technical_signals"] = (
        "\n".join(all_signals) if all_signals else "无明确技术信号"
    )


# ════════════════════════════════════════════════════════════
# Tab 4: 投资建议
# ════════════════════════════════════════════════════════════

def render_tab_advice(client):
    """汇总所有分析数据，调用 AI 生成长线投资建议"""
    st.header("💡 投资建议")
    st.caption("基于量化评分 + AI 分析的长线投资视角（3-6个月维度）")

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
            # 包含长线信号
            lt_label = r.get("long_term_label", "")
            lt_note = f" | 长线: {lt_label}" if lt_label else ""
            diag_lines.append(
                f"- {r['name']} ({r['code']}): "
                f"评分 {r['score']} | 盈亏 {r['pnl_pct']}% | "
                f"{r['label']}{lt_note} | {r['suggestion']}"
            )
    portfolio_diagnosis = "\n".join(diag_lines) if diag_lines else "暂无持仓"

    # 预测汇总对比表
    predictions = st.session_state.get("predictions", [])
    valid_preds = [p for p in predictions if p.pred_1m is not None]
    if valid_preds:
        st.subheader("📊 预测汇总对比（含历史平滑）")

        compare_data = []
        for p in sorted(valid_preds, key=lambda x: x.pred_1m.win_probability, reverse=True):
            smoothed = get_smoothed_prediction(str(p.code))
            compare_data.append({
                "基金": f"{p.name} ({p.code})",
                "近1月概率": f"{p.pred_1m.win_probability:.0%}",
                "近1月预期": f"{p.pred_1m.median_return:+.1%}",
                "近2月概率": f"{p.pred_2m.win_probability:.0%}",
                "近2月预期": f"{p.pred_2m.median_return:+.1%}",
                "近3月概率": f"{p.pred_3m.win_probability:.0%}",
                "近3月预期": f"{p.pred_3m.median_return:+.1%}",
                "置信度": p.confidence,
                "长线信号": ACTION_LABELS.get(
                    smoothed.get("recommended_action", "hold_wait"), "待确认"
                ) if smoothed else "—",
            })
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

        pred_summary_lines = []
        for p in valid_preds:
            smoothed = get_smoothed_prediction(str(p.code))
            action = ACTION_LABELS.get(
                smoothed.get("recommended_action", "hold_wait"), "待确认"
            ) if smoothed else "—"
            pred_summary_lines.append(
                f"{p.name}: 近1月{p.pred_1m.win_probability:.0%}+{p.pred_1m.median_return:+.1%} "
                f"近2月{p.pred_2m.win_probability:.0%}+{p.pred_2m.median_return:+.1%} "
                f"近3月{p.pred_3m.win_probability:.0%}+{p.pred_3m.median_return:+.1%} "
                f"长线信号:{action}"
            )
        technical_signals = technical_signals + "\n\n盈利预测:\n" + "\n".join(pred_summary_lines)

        # AI 预测修正
        if client:
            from analysis.recommender import generate_prediction_analysis
            with st.spinner("AI 正在修正预测..."):
                ai_prediction = generate_prediction_analysis(client, valid_preds, market_sentiment)
            if ai_prediction:
                st.markdown(ai_prediction)

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
