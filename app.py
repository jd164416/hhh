from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv, set_key
from plotly.subplots import make_subplots

from stock_agents.data_source import TushareDataSource
from stock_agents.engine import RetailStockAgentsEngine
from stock_agents.llm_explainer import maybe_explain
from stock_agents.models import DecisionResult
from stock_agents.utils import normalize_cn_code


load_dotenv()
ENV_FILE = Path(__file__).resolve().parent / ".env"


def action_to_cn(action: str) -> str:
    x = action.upper()
    if x == "BUY":
        return "买入"
    if x == "SELL":
        return "减仓/卖出"
    return "观望"


def action_color(action: str) -> str:
    x = action.upper()
    if x == "BUY":
        return "green"
    if x == "SELL":
        return "red"
    return "orange"


def inject_mobile_css() -> None:
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem !important;
                padding-left: 0.8rem !important;
                padding-right: 0.8rem !important;
            }
            [data-testid="stHorizontalBlock"] {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 0.5rem !important;
            }
            [data-testid="stHorizontalBlock"] > div {
                flex: 1 1 calc(50% - 0.5rem) !important;
                min-width: calc(50% - 0.5rem) !important;
            }
        }
        @media (max-width: 480px) {
            [data-testid="stHorizontalBlock"] > div {
                flex-basis: 100% !important;
                min-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_config_state() -> None:
    defaults = {
        "cfg_tushare_token": os.getenv("TUSHARE_TOKEN", ""),
        "cfg_tushare_base_url": os.getenv("TUSHARE_BASE_URL", "http://tushare.xyz"),
        "cfg_deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def persist_config(token: str, base_url: str, deepseek_key: str) -> None:
    if not ENV_FILE.exists():
        ENV_FILE.touch()
    set_key(str(ENV_FILE), "TUSHARE_TOKEN", token, quote_mode="never")
    set_key(str(ENV_FILE), "TUSHARE_BASE_URL", base_url, quote_mode="never")
    set_key(str(ENV_FILE), "DEEPSEEK_API_KEY", deepseek_key, quote_mode="never")

    os.environ["TUSHARE_TOKEN"] = token
    os.environ["TUSHARE_BASE_URL"] = base_url
    os.environ["DEEPSEEK_API_KEY"] = deepseek_key


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def search_stock_options(token: str, base_url: str, keyword: str) -> list[dict]:
    ds = TushareDataSource(token=token, base_url=base_url or None)
    df = ds.search_stocks(keyword=keyword, limit=30)
    return df.to_dict(orient="records")


def make_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["ma5"], mode="lines", name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["ma20"], mode="lines", name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["ma60"], mode="lines", name="MA60"), row=1, col=1)

    fig.add_trace(
        go.Bar(x=df["trade_date"], y=df["vol"], name="成交量"),
        row=2,
        col=1,
    )
    fig.update_layout(
        title="价格与均线",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend_orientation="h",
        height=620,
    )
    return fig


def make_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["macd_diff"], mode="lines", name="DIFF"))
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["macd_dea"], mode="lines", name="DEA"))
    fig.add_trace(go.Bar(x=df["trade_date"], y=df["macd_hist"], name="MACD柱"))
    fig.update_layout(title="MACD", template="plotly_white", height=320)
    return fig


def make_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["rsi14"], mode="lines", name="RSI14"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI", template="plotly_white", height=260, yaxis_range=[0, 100])
    return fig


def show_decision(decision: DecisionResult, latest_close: float) -> None:
    cn_action = action_to_cn(decision.action)
    color = action_color(decision.action)
    st.markdown(
        f"### 当前建议：<span style='color:{color}'>{cn_action}</span>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("综合评分", f"{decision.overall_score:.1f}/100")
    c2.metric("置信度", f"{decision.confidence:.1f}%")

    c3, c4 = st.columns(2)
    c3.metric("建议仓位", f"{decision.suggested_position_pct:.0f}%")
    c4.metric("最新价", f"{latest_close:.2f}")

    c5, c6 = st.columns(2)
    c5.metric("止损位", f"{decision.stop_loss:.2f}")
    c6.metric("止盈位", f"{decision.take_profit:.2f}")

    st.caption(decision.summary)


def main() -> None:
    st.set_page_config(
        page_title="散户股票 Agents",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_mobile_css()
    ensure_config_state()

    st.title("散户股票 Agents")
    st.caption("Tushare 数据 + 多 Agent 评分 + DeepSeek 中文解释")

    with st.sidebar:
        st.header("参数")
        token = st.text_input(
            "Tushare Token",
            key="cfg_tushare_token",
            type="password",
        )
        tushare_base_url = st.text_input(
            "Tushare API URL",
            key="cfg_tushare_base_url",
            help="若默认地址可用，保持不变即可。",
        )
        deepseek_key = st.text_input(
            "DeepSeek API Key（可选）",
            key="cfg_deepseek_api_key",
            type="password",
        )

        if st.button("保存配置（刷新后保留）"):
            try:
                persist_config(
                    token=token.strip(),
                    base_url=tushare_base_url.strip() or "http://tushare.xyz",
                    deepseek_key=deepseek_key.strip(),
                )
                st.success("配置已保存。")
            except Exception as exc:
                st.error(f"配置保存失败：{exc}")

        raw_code = st.text_input("股票代码", value="600519")
        search_keyword = st.text_input("名称搜索（中文名/代码）", value="")
        selected_ts_code = ""
        selected_name = ""
        if token.strip() and search_keyword.strip():
            try:
                options = search_stock_options(
                    token=token.strip(),
                    base_url=tushare_base_url.strip(),
                    keyword=search_keyword.strip(),
                )
                if options:
                    option_labels = [
                        f"{o['name']} ({o['ts_code']})"
                        + (f" | {o.get('industry', '')}" if o.get("industry") else "")
                        for o in options
                    ]
                    picked = st.selectbox("搜索结果", options=option_labels, index=0)
                    idx = option_labels.index(picked)
                    selected_ts_code = str(options[idx]["ts_code"])
                    selected_name = str(options[idx]["name"])
                else:
                    st.caption("未找到匹配股票。")
            except Exception as exc:
                st.caption(f"名称搜索失败：{exc}")

        trade_date = st.date_input("分析日期", value=date.today() - timedelta(days=1))
        lookback_days = st.slider("回看天数", min_value=120, max_value=500, value=260, step=20)
        run = st.button("开始分析", type="primary")

    st.info("仅用于研究与学习，不构成投资建议。")

    if not run:
        return

    if not token:
        st.error("请先填写 Tushare Token。")
        return

    if selected_ts_code:
        ts_code = selected_ts_code
    else:
        try:
            ts_code = normalize_cn_code(raw_code)
        except ValueError as exc:
            st.error(str(exc))
            return

    with st.spinner("正在拉取数据并运行 Agents，请稍候..."):
        try:
            engine = RetailStockAgentsEngine(
                tushare_token=token,
                tushare_base_url=tushare_base_url.strip() or None,
            )
            output = engine.analyze(
                ts_code=ts_code,
                trade_date=trade_date.strftime("%Y-%m-%d"),
                lookback_days=lookback_days,
            )
            stock_name = selected_name or engine.data_source.get_stock_name(ts_code)
        except Exception as exc:
            st.error(f"分析失败：{exc}")
            return

    enriched = output.enriched_df.tail(180).copy()
    latest = output.enriched_df.iloc[-1]
    decision = output.decision

    st.subheader(f"标的：{stock_name} ({ts_code})")
    show_decision(decision, float(latest["close"]))

    st.subheader("Agent 打分")
    report_rows = [{"Agent": r.name, "分数": round(r.score, 1), "结论": r.summary} for r in decision.reports]
    score_df = pd.DataFrame(report_rows)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.subheader("图形分析")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(make_price_chart(enriched), use_container_width=True)
    with c2:
        st.plotly_chart(make_macd_chart(enriched), use_container_width=True)
        st.plotly_chart(make_rsi_chart(enriched), use_container_width=True)

    st.subheader("DeepSeek 中文解释")
    explanation = maybe_explain(
        api_key=deepseek_key,
        ticker=ts_code,
        trade_date=trade_date.strftime("%Y-%m-%d"),
        decision=decision,
    )
    st.write(explanation)


if __name__ == "__main__":
    main()
