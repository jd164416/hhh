from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import tushare as ts
import tushare.pro.client as ts_client

try:
    import akshare as ak
except Exception:
    ak = None

from stock_agents.data_source import build_data_source
from stock_agents.engine import RetailStockAgentsEngine
from stock_agents.llm_explainer import maybe_explain
from stock_agents.models import DecisionResult


load_dotenv()
ENV_FILE = Path(__file__).resolve().parent / ".env"
ENV_KEYS = [
    "DATA_SOURCE",
    "TUSHARE_TOKEN",
    "TUSHARE_BASE_URL",
    "LLM_PROVIDER",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
]


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
        "cfg_data_source": os.getenv("DATA_SOURCE", "auto"),
        "cfg_tushare_token": os.getenv("TUSHARE_TOKEN", ""),
        "cfg_tushare_base_url": os.getenv("TUSHARE_BASE_URL", "http://tushare.xyz"),
        "cfg_llm_provider": os.getenv("LLM_PROVIDER", "deepseek"),
        "cfg_deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "cfg_gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def persist_config(
    data_source_mode: str,
    token: str,
    base_url: str,
    llm_provider: str,
    deepseek_key: str,
    gemini_key: str,
) -> None:
    pairs: dict[str, str] = {}
    if ENV_FILE.exists():
        for raw_line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip("'").strip('"')
            pairs[key] = value

    pairs["DATA_SOURCE"] = data_source_mode
    pairs["TUSHARE_TOKEN"] = token
    pairs["TUSHARE_BASE_URL"] = base_url
    pairs["LLM_PROVIDER"] = llm_provider
    pairs["DEEPSEEK_API_KEY"] = deepseek_key
    pairs["GEMINI_API_KEY"] = gemini_key

    normalized = "\n".join(f"{k}={pairs.get(k, '')}" for k in ENV_KEYS) + "\n"
    ENV_FILE.write_text(normalized, encoding="utf-8")

    os.environ["DATA_SOURCE"] = data_source_mode
    os.environ["TUSHARE_TOKEN"] = token
    os.environ["TUSHARE_BASE_URL"] = base_url
    os.environ["LLM_PROVIDER"] = llm_provider
    os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    os.environ["GEMINI_API_KEY"] = gemini_key


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def search_stock_options(
    data_source_mode: str,
    token: str,
    base_url: str,
    keyword: str,
) -> list[dict]:
    ds = build_data_source(
        data_source_mode=data_source_mode,
        tushare_token=token,
        tushare_base_url=base_url or None,
    )
    df = ds.search_stocks(keyword=keyword, limit=30)
    return df.to_dict(orient="records")


def _profile_template(ts_code: str, stock_name: str, source: str) -> dict[str, str]:
    return {
        "名称": stock_name or ts_code,
        "代码": ts_code,
        "数据源": source,
        "行业": "-",
        "地区": "-",
        "上市日期": "-",
        "主营业务": "-",
        "经营范围": "-",
        "法定代表人": "-",
        "员工人数": "-",
        "报告期": "-",
        "营业收入": "-",
        "归母净利润": "-",
        "总资产": "-",
        "总负债": "-",
        "ROE": "-",
        "毛利率": "-",
        "净利率": "-",
        "资产负债率": "-",
    }


def _fmt_date_yyyymmdd(x: object) -> str:
    s = str(x or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s if s else "-"


def _fmt_yi(x: object) -> str:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return "-"
    return f"{float(v) / 1e8:.2f}亿"


def _fmt_pct(x: object) -> str:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return "-"
    return f"{float(v):.2f}%"


def _fetch_profile_tushare(token: str, base_url: str, ts_code: str, stock_name: str) -> dict[str, str]:
    if base_url:
        ts_client.DataApi._DataApi__http_url = base_url.rstrip("/")
    ts.set_token(token)
    pro = ts.pro_api(token)
    profile = _profile_template(ts_code=ts_code, stock_name=stock_name, source="Tushare")

    try:
        base_frames = []
        for status in ("L", "P", "D"):
            df = pro.stock_basic(exchange="", list_status=status, fields="ts_code,name,area,industry,list_date")
            if df is not None and not df.empty:
                base_frames.append(df)
        if base_frames:
            base = pd.concat(base_frames, ignore_index=True).drop_duplicates(subset=["ts_code"])
            row = base.loc[base["ts_code"].str.upper() == ts_code.upper()]
            if not row.empty:
                r = row.iloc[0]
                profile["名称"] = str(r.get("name", profile["名称"]))
                profile["行业"] = str(r.get("industry", "-") or "-")
                profile["地区"] = str(r.get("area", "-") or "-")
                profile["上市日期"] = _fmt_date_yyyymmdd(r.get("list_date"))
    except Exception:
        pass

    try:
        exchange = "SSE" if ts_code.endswith(".SH") else "SZSE"
        comp = pro.stock_company(
            exchange=exchange,
            fields="ts_code,chairman,employees,main_business,business_scope",
        )
        if comp is not None and not comp.empty:
            row = comp.loc[comp["ts_code"].str.upper() == ts_code.upper()]
            if not row.empty:
                r = row.iloc[0]
                profile["法定代表人"] = str(r.get("chairman", "-") or "-")
                profile["员工人数"] = str(r.get("employees", "-") or "-")
                profile["主营业务"] = str(r.get("main_business", "-") or "-")
                profile["经营范围"] = str(r.get("business_scope", "-") or "-")
    except Exception:
        pass

    report_period = ""
    try:
        income = pro.income(ts_code=ts_code, fields="ts_code,end_date,total_revenue,n_income")
        if income is not None and not income.empty:
            x = income.copy()
            x["end_date"] = x["end_date"].astype(str)
            x = x.sort_values("end_date", ascending=False)
            annual = x[x["end_date"].str.endswith("1231")]
            r = annual.iloc[0] if not annual.empty else x.iloc[0]
            report_period = str(r.get("end_date", ""))
            profile["报告期"] = _fmt_date_yyyymmdd(report_period)
            profile["营业收入"] = _fmt_yi(r.get("total_revenue"))
            profile["归母净利润"] = _fmt_yi(r.get("n_income"))
    except Exception:
        pass

    try:
        bal = pro.balancesheet(ts_code=ts_code, fields="ts_code,end_date,total_assets,total_liab")
        if bal is not None and not bal.empty:
            y = bal.copy()
            y["end_date"] = y["end_date"].astype(str)
            y = y.sort_values("end_date", ascending=False)
            r = y.iloc[0]
            if report_period:
                z = y[y["end_date"] == report_period]
                if not z.empty:
                    r = z.iloc[0]
            profile["总资产"] = _fmt_yi(r.get("total_assets"))
            profile["总负债"] = _fmt_yi(r.get("total_liab"))
    except Exception:
        pass

    try:
        fi = pro.fina_indicator(
            ts_code=ts_code,
            fields="ts_code,end_date,roe,grossprofit_margin,netprofit_margin,debt_to_assets",
        )
        if fi is not None and not fi.empty:
            f = fi.copy()
            f["end_date"] = f["end_date"].astype(str)
            f = f.sort_values("end_date", ascending=False)
            r = f.iloc[0]
            if report_period:
                z = f[f["end_date"] == report_period]
                if not z.empty:
                    r = z.iloc[0]
            profile["ROE"] = _fmt_pct(r.get("roe"))
            profile["毛利率"] = _fmt_pct(r.get("grossprofit_margin"))
            profile["净利率"] = _fmt_pct(r.get("netprofit_margin"))
            profile["资产负债率"] = _fmt_pct(r.get("debt_to_assets"))
            if profile["报告期"] == "-":
                profile["报告期"] = _fmt_date_yyyymmdd(r.get("end_date"))
    except Exception:
        pass

    return profile


def _fetch_profile_akshare(ts_code: str, stock_name: str) -> dict[str, str]:
    if ak is None:
        raise RuntimeError("未安装 AkShare。")
    code6 = ts_code.split(".")[0]
    profile = _profile_template(ts_code=ts_code, stock_name=stock_name, source="AkShare")

    try:
        info = ak.stock_individual_info_em(symbol=code6)
    except Exception:
        info = pd.DataFrame()

    info_map: dict[str, str] = {}
    if info is not None and not info.empty and info.shape[1] >= 2:
        c0 = info.columns[0]
        c1 = info.columns[1]
        for _, r in info.iterrows():
            k = str(r.get(c0, "")).strip()
            v = str(r.get(c1, "")).strip()
            if k:
                info_map[k] = v

    profile["行业"] = info_map.get("行业", info_map.get("所属行业", profile["行业"]))
    profile["上市日期"] = _fmt_date_yyyymmdd(info_map.get("上市时间", info_map.get("上市日期", "-")))
    profile["主营业务"] = info_map.get("主营业务", info_map.get("公司简介", "-"))
    profile["经营范围"] = info_map.get("经营范围", info_map.get("公司简介", "-"))
    profile["法定代表人"] = info_map.get("法人代表", info_map.get("法定代表人", "-"))

    try:
        fin = ak.stock_financial_analysis_indicator(symbol=code6)
    except Exception:
        fin = pd.DataFrame()

    if fin is not None and not fin.empty:
        date_col = next((c for c in ["日期", "报告期", "end_date"] if c in fin.columns), None)
        if date_col is not None:
            fin[date_col] = pd.to_datetime(fin[date_col], errors="coerce")
            fin = fin.sort_values(date_col, ascending=False)
        r = fin.iloc[0]
        if date_col is not None and pd.notna(r.get(date_col)):
            profile["报告期"] = pd.to_datetime(r[date_col]).strftime("%Y-%m-%d")

        for key, targets in {
            "ROE": ["净资产收益率(%)", "ROE(%)", "ROE"],
            "毛利率": ["销售毛利率(%)", "毛利率(%)", "毛利率"],
            "净利率": ["销售净利率(%)", "净利率(%)", "净利率"],
            "资产负债率": ["资产负债率(%)", "资产负债率"],
        }.items():
            col = next((c for c in targets if c in fin.columns), None)
            if col:
                profile[key] = _fmt_pct(r.get(col))

    return profile


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_stock_profile_data(
    data_source_mode: str,
    token: str,
    base_url: str,
    ts_code: str,
    stock_name: str,
) -> dict[str, str]:
    mode = (data_source_mode or "auto").strip().lower()
    if mode in ("tushare", "auto") and token.strip():
        try:
            return _fetch_profile_tushare(token=token.strip(), base_url=base_url.strip(), ts_code=ts_code, stock_name=stock_name)
        except Exception:
            if mode == "tushare":
                raise
    return _fetch_profile_akshare(ts_code=ts_code, stock_name=stock_name)


def render_stock_profile(profile: dict[str, str]) -> None:
    st.subheader("个股介绍")
    if not profile:
        st.warning("暂未获取到个股介绍信息。")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行业", str(profile.get("行业", "-")))
    c2.metric("地区", str(profile.get("地区", "-")))
    c3.metric("上市日期", str(profile.get("上市日期", "-")))
    c4.metric("数据源", str(profile.get("数据源", "-")))

    st.markdown(f"**主营业务**：{profile.get('主营业务', '-')}")
    st.markdown(f"**经营范围**：{profile.get('经营范围', '-')}")

    ext = []
    if profile.get("法定代表人", "-") not in ("", "-"):
        ext.append(f"法定代表人：{profile.get('法定代表人')}")
    if profile.get("员工人数", "-") not in ("", "-"):
        ext.append(f"员工人数：{profile.get('员工人数')}")
    if ext:
        st.caption(" | ".join(ext))

    st.subheader("财务概览")
    rows = [
        {"指标": "报告期", "数值": profile.get("报告期", "-")},
        {"指标": "营业收入", "数值": profile.get("营业收入", "-")},
        {"指标": "归母净利润", "数值": profile.get("归母净利润", "-")},
        {"指标": "总资产", "数值": profile.get("总资产", "-")},
        {"指标": "总负债", "数值": profile.get("总负债", "-")},
        {"指标": "ROE", "数值": profile.get("ROE", "-")},
        {"指标": "毛利率", "数值": profile.get("毛利率", "-")},
        {"指标": "净利率", "数值": profile.get("净利率", "-")},
        {"指标": "资产负债率", "数值": profile.get("资产负债率", "-")},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
    st.caption("Tushare/AkShare 数据 + 多 Agent 评分 + DeepSeek/Gemini 中文解释")

    with st.sidebar:
        st.header("参数")
        source_options = ["auto", "tushare", "akshare"]
        source_default = st.session_state["cfg_data_source"] if st.session_state["cfg_data_source"] in source_options else "auto"
        data_source_mode = st.selectbox(
            "数据源",
            options=source_options,
            index=source_options.index(source_default),
            format_func=lambda x: {"auto": "自动(优先 Tushare)", "tushare": "仅 Tushare", "akshare": "仅 AkShare"}[x],
        )
        st.session_state["cfg_data_source"] = data_source_mode
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
        llm_provider = st.selectbox(
            "解释模型提供商",
            options=["deepseek", "gemini"],
            index=0 if st.session_state["cfg_llm_provider"] == "deepseek" else 1,
            format_func=lambda x: "DeepSeek" if x == "deepseek" else "Gemini",
        )
        st.session_state["cfg_llm_provider"] = llm_provider
        gemini_key = st.text_input(
            "Gemini API Key（可选）",
            key="cfg_gemini_api_key",
            type="password",
        )

        if st.button("保存配置（刷新后保留）"):
            try:
                persist_config(
                    data_source_mode=data_source_mode,
                    token=token.strip(),
                    base_url=tushare_base_url.strip() or "http://tushare.xyz",
                    llm_provider=llm_provider,
                    deepseek_key=deepseek_key.strip(),
                    gemini_key=gemini_key.strip(),
                )
                st.success("配置已保存。")
            except Exception as exc:
                st.error(f"配置保存失败：{exc}")

        search_keyword = st.text_input("名称/代码搜索", value="600519")
        selected_ts_code = ""
        selected_name = ""
        if search_keyword.strip():
            if data_source_mode == "tushare" and not token.strip():
                st.caption("仅 Tushare 模式需要先填写 Tushare Token。")
            else:
                try:
                    options = search_stock_options(
                        data_source_mode=data_source_mode,
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
                        st.caption("未找到匹配股票，请输入股票名称或6位代码。")
                except Exception as exc:
                    st.caption(f"名称搜索失败：{exc}")

        trade_date = st.date_input("分析日期", value=date.today())
        lookback_days = st.slider("回看天数", min_value=120, max_value=500, value=260, step=20)
        run = st.button("开始分析", type="primary")

    st.info("仅用于研究与学习，不构成投资建议。")

    if not run:
        return

    if data_source_mode == "tushare" and not token:
        st.error("仅 Tushare 模式需要填写 Tushare Token。")
        return

    if not search_keyword.strip():
        st.error("请先输入股票名称或代码。")
        return
    if not selected_ts_code:
        st.error("未找到可分析的股票，请修改关键词并从搜索结果选择。")
        return
    ts_code = selected_ts_code

    with st.spinner("正在拉取数据并运行 Agents，请稍候..."):
        try:
            engine = RetailStockAgentsEngine(
                tushare_token=token,
                tushare_base_url=tushare_base_url.strip() or None,
                data_source_mode=data_source_mode,
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

    latest = output.enriched_df.iloc[-1]
    decision = output.decision
    try:
        profile = fetch_stock_profile_data(
            data_source_mode=data_source_mode,
            token=token,
            base_url=tushare_base_url.strip() or "http://tushare.xyz",
            ts_code=ts_code,
            stock_name=stock_name,
        )
    except Exception:
        profile = {}

    st.subheader(f"标的：{stock_name} ({ts_code})")
    if output.used_trade_date != trade_date.strftime("%Y-%m-%d"):
        st.warning(
            f"你选择的日期是 {trade_date.strftime('%Y-%m-%d')}，"
            f"该日可能未收盘/非交易日，当前已自动使用最新交易日 {output.used_trade_date}。"
        )
    else:
        st.caption(f"当前使用交易日：{output.used_trade_date}")
    show_decision(decision, float(latest["close"]))

    st.subheader("Agent 打分")
    report_rows = [{"Agent": r.name, "分数": round(r.score, 1), "结论": r.summary} for r in decision.reports]
    score_df = pd.DataFrame(report_rows)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    render_stock_profile(profile)

    st.subheader("LLM 中文解释")
    explanation = maybe_explain(
        provider=llm_provider,
        deepseek_api_key=deepseek_key,
        gemini_api_key=gemini_key,
        ticker=ts_code,
        trade_date=trade_date.strftime("%Y-%m-%d"),
        decision=decision,
    )
    st.write(explanation)


if __name__ == "__main__":
    main()
