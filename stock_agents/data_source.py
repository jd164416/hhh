from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Protocol

import pandas as pd
import tushare as ts
import tushare.pro.client as ts_client

from .utils import to_tushare_date

try:
    import akshare as ak
except Exception:  # pragma: no cover
    ak = None


@dataclass
class MarketDataBundle:
    daily: pd.DataFrame
    daily_basic: pd.DataFrame
    moneyflow: pd.DataFrame


class StockDataSource(Protocol):
    def fetch_bundle(self, ts_code: str, analysis_date: str, lookback_days: int = 260) -> MarketDataBundle:
        ...

    def fetch_news_sentiment(
        self,
        ts_code: str,
        stock_name: str,
        analysis_date: str | None = None,
        lookback_days: int = 7,
        limit: int = 20,
    ) -> pd.DataFrame:
        ...

    def search_stocks(self, keyword: str, limit: int = 30) -> pd.DataFrame:
        ...

    def get_stock_name(self, ts_code: str) -> str:
        ...


class TushareDataSource:
    def __init__(self, token: str, base_url: str | None = None):
        if not token:
            raise ValueError("缺少 Tushare Token，请先配置 TUSHARE_TOKEN。")

        if base_url:
            # For environments that require custom endpoint, e.g. http://tushare.xyz
            ts_client.DataApi._DataApi__http_url = base_url.rstrip("/")

        ts.set_token(token)
        self.pro = ts.pro_api(token)
        self._stock_basic_cache: pd.DataFrame | None = None

    def fetch_bundle(
        self,
        ts_code: str,
        analysis_date: str,
        lookback_days: int = 260,
    ) -> MarketDataBundle:
        end_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days * 2)

        start_date = to_tushare_date(start_dt)
        end_date = to_tushare_date(end_dt)

        daily = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        daily_basic = self.pro.daily_basic(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        moneyflow = self.pro.moneyflow(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

        if daily is None or daily.empty:
            raise RuntimeError(f"未获取到 {ts_code} 的日线数据，请检查代码或日期。")

        return MarketDataBundle(
            daily=self._normalize_daily(daily),
            daily_basic=self._normalize_generic(daily_basic),
            moneyflow=self._normalize_generic(moneyflow),
        )

    def fetch_news_sentiment(
        self,
        ts_code: str,
        stock_name: str,
        analysis_date: str | None = None,
        lookback_days: int = 7,
        limit: int = 20,
    ) -> pd.DataFrame:
        if analysis_date:
            end_dt = datetime.strptime(analysis_date, "%Y-%m-%d") + timedelta(hours=23, minutes=59, seconds=59)
        else:
            end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days)
        code6 = ts_code.split(".")[0]

        news = pd.DataFrame()
        try:
            news = self.pro.news(
                start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception:
            news = pd.DataFrame()

        # Widen the window when the initial query is empty.
        if news is None or news.empty:
            wide_start = end_dt - timedelta(days=max(30, lookback_days))
            try:
                news = self.pro.news(
                    start_date=wide_start.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                )
            except Exception:
                news = pd.DataFrame()

        if news is None or news.empty:
            # Final fallback: generic market news via AkShare, to avoid empty UI.
            return self._fallback_market_news_from_ak(
                start_dt=end_dt - timedelta(days=max(30, lookback_days)),
                end_dt=end_dt,
                limit=limit,
            )

        df = news.copy()
        for col in ["datetime", "title", "content", "src"]:
            if col not in df.columns:
                df[col] = ""

        text = (df["title"].astype(str).fillna("") + " " + df["content"].astype(str).fillna("")).str.lower()
        keywords = self._build_news_keywords(stock_name=stock_name, code6=code6)
        mask = pd.Series(False, index=df.index)
        for key in keywords:
            if key:
                mask = mask | text.str.contains(re.escape(key.lower()), na=False)
        df = df.loc[mask].copy()
        if df.empty:
            fallback = news.copy()
            for col in ["datetime", "title", "content", "src"]:
                if col not in fallback.columns:
                    fallback[col] = ""
            fallback["datetime"] = pd.to_datetime(fallback["datetime"], errors="coerce")
            fallback = fallback.sort_values("datetime", ascending=False).head(limit)
            fallback_text = (fallback["title"].astype(str).fillna("") + " " + fallback["content"].astype(str).fillna("")).str.lower()
            fallback["sentiment_score"] = fallback_text.apply(self._score_text_sentiment).astype(float)
            fallback["sentiment_label"] = fallback["sentiment_score"].apply(self._label_sentiment)
            fallback["source"] = fallback["src"].astype(str)
            fallback["match_type"] = "市场新闻(未精准匹配)"
            keep_cols = ["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
            return fallback[keep_cols].reset_index(drop=True)

        df["sentiment_score"] = text.loc[df.index].apply(self._score_text_sentiment).astype(float)
        df["sentiment_label"] = df["sentiment_score"].apply(self._label_sentiment)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime", ascending=False).head(limit)
        df["source"] = df["src"].astype(str)
        df["match_type"] = "个股匹配"

        keep_cols = ["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
        return df[keep_cols].reset_index(drop=True)

    def _fallback_market_news_from_ak(
        self,
        start_dt: datetime,
        end_dt: datetime,
        limit: int,
    ) -> pd.DataFrame:
        if ak is None:
            return self._empty_news_frame()
        try:
            raw = ak.stock_info_global_sina()
        except Exception:
            return self._empty_news_frame()
        if raw is None or raw.empty:
            return self._empty_news_frame()

        time_col = next((c for c in ["时间", "time", "datetime", "date"] if c in raw.columns), None)
        text_col = next((c for c in ["内容", "title", "标题", "content"] if c in raw.columns), None)
        source_col = next((c for c in ["来源", "source"] if c in raw.columns), None)
        if text_col is None:
            return self._empty_news_frame()

        df = pd.DataFrame()
        if time_col is not None:
            df["datetime"] = pd.to_datetime(raw[time_col], errors="coerce")
        else:
            df["datetime"] = pd.Timestamp.now()
        df["title"] = raw[text_col].astype(str)
        df["source"] = raw[source_col].astype(str) if source_col else "sina_global"
        df = df.dropna(subset=["datetime"]).copy()
        filtered = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
        if filtered.empty:
            filtered = df
        filtered = filtered.sort_values("datetime", ascending=False).head(limit).copy()
        filtered["sentiment_score"] = filtered["title"].apply(self._score_text_sentiment).astype(float)
        filtered["sentiment_label"] = filtered["sentiment_score"].apply(self._label_sentiment)
        filtered["match_type"] = "市场新闻(兜底)"
        return filtered[["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]].reset_index(drop=True)

    def search_stocks(self, keyword: str, limit: int = 30) -> pd.DataFrame:
        key = (keyword or "").strip().lower()
        if not key:
            return pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry"])

        base = self._get_stock_basic()
        if base.empty:
            return base

        mask = (
            base["name"].str.lower().str.contains(key, na=False)
            | base["symbol"].str.lower().str.contains(key, na=False)
            | base["ts_code"].str.lower().str.contains(key, na=False)
        )
        out = base.loc[mask].copy()
        return out.head(limit).reset_index(drop=True)

    def get_stock_name(self, ts_code: str) -> str:
        base = self._get_stock_basic()
        if base.empty:
            return ts_code
        row = base.loc[base["ts_code"].str.upper() == ts_code.upper()]
        if row.empty:
            return ts_code
        return str(row.iloc[0]["name"])

    def _get_stock_basic(self) -> pd.DataFrame:
        if self._stock_basic_cache is not None:
            return self._stock_basic_cache

        frames = []
        for list_status in ("L", "P", "D"):
            df = self.pro.stock_basic(
                exchange="",
                list_status=list_status,
                fields="ts_code,symbol,name,area,industry,list_date",
            )
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            self._stock_basic_cache = pd.DataFrame(
                columns=["ts_code", "symbol", "name", "area", "industry", "list_date"]
            )
        else:
            merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"])
            self._stock_basic_cache = merged.sort_values("ts_code").reset_index(drop=True)

        return self._stock_basic_cache

    @staticmethod
    def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["trade_date"] = pd.to_datetime(x["trade_date"], format="%Y%m%d")
        x = x.sort_values("trade_date").reset_index(drop=True)
        numeric_cols = ["open", "high", "low", "close", "pre_close", "vol", "amount", "pct_chg"]
        for c in numeric_cols:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        return x

    @staticmethod
    def _normalize_generic(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["trade_date"])
        x = df.copy()
        x["trade_date"] = pd.to_datetime(x["trade_date"], format="%Y%m%d")
        x = x.sort_values("trade_date").reset_index(drop=True)
        return x

    @staticmethod
    def _empty_news_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
        )

    @staticmethod
    def _score_text_sentiment(text: str) -> float:
        positive_words = ["增长", "利好", "突破", "上调", "回购", "超预期", "盈利", "签约", "中标", "看多"]
        negative_words = ["下滑", "利空", "暴跌", "下调", "亏损", "减持", "违约", "处罚", "看空", "风险"]

        score = 0.0
        txt = str(text)
        for word in positive_words:
            if word in txt:
                score += 1.0
        for word in negative_words:
            if word in txt:
                score -= 1.0
        return score

    @staticmethod
    def _label_sentiment(score: float) -> str:
        if score > 0:
            return "偏多"
        if score < 0:
            return "偏空"
        return "中性"

    @staticmethod
    def _build_news_keywords(stock_name: str, code6: str) -> list[str]:
        name = (stock_name or "").strip()
        aliases = {name, code6}
        cleaned = (
            name.replace("股份有限公司", "")
            .replace("股份", "")
            .replace("集团", "")
            .replace("有限", "")
            .strip()
        )
        if cleaned:
            aliases.add(cleaned)
        if len(cleaned) >= 2:
            aliases.add(cleaned[:2])
        if len(cleaned) >= 3:
            aliases.add(cleaned[:3])
        return [x for x in aliases if x]


class AkShareDataSource:
    def __init__(self):
        if ak is None:
            raise RuntimeError("未安装 akshare，请先执行 pip install akshare。")
        self._stock_basic_cache: pd.DataFrame | None = None
        self._spot_cache: pd.DataFrame | None = None

    def fetch_bundle(
        self,
        ts_code: str,
        analysis_date: str,
        lookback_days: int = 260,
    ) -> MarketDataBundle:
        end_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days * 2)
        code6 = ts_code.split(".")[0]

        daily_raw = ak.stock_zh_a_hist(
            symbol=code6,
            period="daily",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
            adjust="",
        )
        if daily_raw is None or daily_raw.empty:
            raise RuntimeError(f"未获取到 {ts_code} 的日线数据，请检查代码或日期。")

        daily = self._normalize_daily_ak(daily_raw, ts_code=ts_code)
        if daily.empty:
            raise RuntimeError(f"{ts_code} 日线数据为空。")

        daily_basic = self._build_daily_basic_ak(daily, code6=code6)
        moneyflow = pd.DataFrame(columns=["trade_date"])
        return MarketDataBundle(
            daily=daily,
            daily_basic=daily_basic,
            moneyflow=moneyflow,
        )

    def fetch_news_sentiment(
        self,
        ts_code: str,
        stock_name: str,
        analysis_date: str | None = None,
        lookback_days: int = 7,
        limit: int = 20,
    ) -> pd.DataFrame:
        code6 = ts_code.split(".")[0]
        if analysis_date:
            end_dt = datetime.strptime(analysis_date, "%Y-%m-%d") + timedelta(hours=23, minutes=59, seconds=59)
        else:
            end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days)

        try:
            raw = ak.stock_news_em(symbol=code6)
        except Exception:
            return self._fallback_market_news_from_ak(start_dt=start_dt, end_dt=end_dt, limit=limit)

        if raw is None or raw.empty:
            return self._fallback_market_news_from_ak(start_dt=start_dt, end_dt=end_dt, limit=limit)

        df = raw.copy()
        datetime_col = self._pick_col(df, ["发布时间", "时间", "datetime", "date"])
        title_col = self._pick_col(df, ["新闻标题", "标题", "title"])
        content_col = self._pick_col(df, ["新闻内容", "内容", "content"])
        source_col = self._pick_col(df, ["文章来源", "来源", "source"])

        if datetime_col is None or title_col is None:
            return self._empty_news_frame()

        norm = pd.DataFrame()
        norm["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
        norm["title"] = df[title_col].astype(str)
        norm["content"] = df[content_col].astype(str) if content_col else ""
        norm["source"] = df[source_col].astype(str) if source_col else "akshare"
        norm = norm.dropna(subset=["datetime"]).copy()
        in_window = norm[(norm["datetime"] >= start_dt) & (norm["datetime"] <= end_dt)]
        if in_window.empty:
            # Expand window to reduce "no news matched" cases.
            wide_start = end_dt - timedelta(days=max(30, lookback_days))
            in_window = norm[(norm["datetime"] >= wide_start) & (norm["datetime"] <= end_dt)]
        if in_window.empty:
            # If still empty by date, keep latest records as fallback.
            in_window = norm.sort_values("datetime", ascending=False).head(limit).copy()

        norm = in_window.copy()

        text = (norm["title"].fillna("") + " " + norm["content"].fillna("")).str.lower()
        keywords = self._build_news_keywords(stock_name=stock_name, code6=code6)
        mask = pd.Series(False, index=norm.index)
        for key in keywords:
            if key:
                mask = mask | text.str.contains(re.escape(key.lower()), na=False)
        matched = norm.loc[mask].copy()

        if matched.empty:
            fallback = norm.sort_values("datetime", ascending=False).head(limit).copy()
            fallback_text = (fallback["title"].fillna("") + " " + fallback["content"].fillna("")).str.lower()
            fallback["sentiment_score"] = fallback_text.apply(self._score_text_sentiment).astype(float)
            fallback["sentiment_label"] = fallback["sentiment_score"].apply(self._label_sentiment)
            fallback["match_type"] = "市场新闻(未精准匹配)"
            keep_cols = ["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
            return fallback[keep_cols].reset_index(drop=True)

        matched["sentiment_score"] = (
            (matched["title"].fillna("") + " " + matched["content"].fillna("")).str.lower().apply(self._score_text_sentiment).astype(float)
        )
        matched["sentiment_label"] = matched["sentiment_score"].apply(self._label_sentiment)
        matched["match_type"] = "个股匹配"
        matched = matched.sort_values("datetime", ascending=False).head(limit)

        keep_cols = ["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
        return matched[keep_cols].reset_index(drop=True)

    def _fallback_market_news_from_ak(
        self,
        start_dt: datetime,
        end_dt: datetime,
        limit: int,
    ) -> pd.DataFrame:
        try:
            raw = ak.stock_info_global_sina()
        except Exception:
            return self._empty_news_frame()
        if raw is None or raw.empty:
            return self._empty_news_frame()

        time_col = self._pick_col(raw, ["时间", "time", "datetime", "date"])
        text_col = self._pick_col(raw, ["内容", "title", "标题", "content"])
        source_col = self._pick_col(raw, ["来源", "source"])
        if text_col is None:
            return self._empty_news_frame()

        df = pd.DataFrame()
        if time_col is not None:
            df["datetime"] = pd.to_datetime(raw[time_col], errors="coerce")
        else:
            df["datetime"] = pd.Timestamp.now()
        df["title"] = raw[text_col].astype(str)
        df["source"] = raw[source_col].astype(str) if source_col else "sina_global"
        df = df.dropna(subset=["datetime"]).copy()
        filtered = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
        if filtered.empty:
            filtered = df
        filtered = filtered.sort_values("datetime", ascending=False).head(limit).copy()
        filtered["sentiment_score"] = filtered["title"].apply(self._score_text_sentiment).astype(float)
        filtered["sentiment_label"] = filtered["sentiment_score"].apply(self._label_sentiment)
        filtered["match_type"] = "市场新闻(兜底)"
        return filtered[["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]].reset_index(drop=True)

    def search_stocks(self, keyword: str, limit: int = 30) -> pd.DataFrame:
        key = (keyword or "").strip().lower()
        if not key:
            return pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry"])

        base = self._get_stock_basic()
        if base.empty:
            return base

        mask = (
            base["name"].str.lower().str.contains(key, na=False)
            | base["symbol"].str.lower().str.contains(key, na=False)
            | base["ts_code"].str.lower().str.contains(key, na=False)
        )
        return base.loc[mask].head(limit).reset_index(drop=True)

    def get_stock_name(self, ts_code: str) -> str:
        base = self._get_stock_basic()
        if base.empty:
            return ts_code
        row = base.loc[base["ts_code"].str.upper() == ts_code.upper()]
        if row.empty:
            return ts_code
        return str(row.iloc[0]["name"])

    def _build_daily_basic_ak(self, daily: pd.DataFrame, code6: str) -> pd.DataFrame:
        out = pd.DataFrame({"trade_date": daily["trade_date"]})
        if "turnover_rate" in daily.columns:
            out["turnover_rate"] = pd.to_numeric(daily["turnover_rate"], errors="coerce")
        else:
            out["turnover_rate"] = pd.NA

        out["pe_ttm"] = pd.NA
        out["pb"] = pd.NA
        spot = self._get_spot_snapshot()
        row = spot.loc[spot["symbol"] == code6]
        if not row.empty:
            pe = pd.to_numeric(row.iloc[0].get("pe_ttm"), errors="coerce")
            pb = pd.to_numeric(row.iloc[0].get("pb"), errors="coerce")
            if pd.notna(pe):
                out["pe_ttm"] = float(pe)
            if pd.notna(pb):
                out["pb"] = float(pb)
        return out.sort_values("trade_date").reset_index(drop=True)

    def _get_stock_basic(self) -> pd.DataFrame:
        if self._stock_basic_cache is not None:
            return self._stock_basic_cache

        spot = self._get_spot_snapshot()
        if not spot.empty:
            base = pd.DataFrame(
                {
                    "ts_code": spot["symbol"].apply(self._code6_to_ts),
                    "symbol": spot["symbol"],
                    "name": spot["name"],
                    "area": "",
                    "industry": "",
                }
            )
            self._stock_basic_cache = base.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
            return self._stock_basic_cache

        try:
            code_name = ak.stock_info_a_code_name()
        except Exception:
            code_name = pd.DataFrame()

        if code_name is None or code_name.empty:
            self._stock_basic_cache = pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry"])
            return self._stock_basic_cache

        col_code = self._pick_col(code_name, ["code", "证券代码", "股票代码", "代码"])
        col_name = self._pick_col(code_name, ["name", "证券简称", "股票简称", "名称"])
        if not col_code or not col_name:
            self._stock_basic_cache = pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry"])
            return self._stock_basic_cache

        base = pd.DataFrame()
        base["symbol"] = code_name[col_code].astype(str).str.zfill(6)
        base["ts_code"] = base["symbol"].apply(self._code6_to_ts)
        base["name"] = code_name[col_name].astype(str)
        base["area"] = ""
        base["industry"] = ""
        self._stock_basic_cache = base.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        return self._stock_basic_cache

    def _get_spot_snapshot(self) -> pd.DataFrame:
        if self._spot_cache is not None:
            return self._spot_cache
        try:
            raw = ak.stock_zh_a_spot_em()
        except Exception:
            raw = pd.DataFrame()

        if raw is None or raw.empty:
            self._spot_cache = pd.DataFrame(columns=["symbol", "name", "turnover_rate", "pe_ttm", "pb"])
            return self._spot_cache

        out = pd.DataFrame()
        out["symbol"] = raw.get("代码", "").astype(str).str.zfill(6)
        out["name"] = raw.get("名称", "").astype(str)
        out["turnover_rate"] = pd.to_numeric(raw.get("换手率"), errors="coerce")
        out["pe_ttm"] = pd.to_numeric(raw.get("市盈率-动态"), errors="coerce")
        out["pb"] = pd.to_numeric(raw.get("市净率"), errors="coerce")
        self._spot_cache = out.dropna(subset=["symbol"]).copy()
        return self._spot_cache

    @staticmethod
    def _normalize_daily_ak(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        x = df.copy()
        rename_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "vol",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "换手率": "turnover_rate",
        }
        x = x.rename(columns=rename_map)
        required = ["trade_date", "open", "high", "low", "close", "vol", "amount"]
        for col in required:
            if col not in x.columns:
                raise RuntimeError(f"AkShare 数据缺少字段: {col}")

        x["trade_date"] = pd.to_datetime(x["trade_date"], errors="coerce")
        x = x.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "vol", "amount", "pct_chg", "turnover_rate"]:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x["pre_close"] = x["close"].shift(1)
        x["ts_code"] = ts_code
        cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount", "pct_chg"]
        if "turnover_rate" in x.columns:
            cols.append("turnover_rate")
        return x[cols]

    @staticmethod
    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _code6_to_ts(code6: str) -> str:
        c = str(code6).strip().zfill(6)
        if c.startswith(("6", "9")):
            return f"{c}.SH"
        return f"{c}.SZ"

    @staticmethod
    def _empty_news_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
        )

    @staticmethod
    def _score_text_sentiment(text: str) -> float:
        positive_words = ["增长", "利好", "突破", "上调", "回购", "超预期", "盈利", "签约", "中标", "看多"]
        negative_words = ["下滑", "利空", "暴跌", "下调", "亏损", "减持", "违约", "处罚", "看空", "风险"]
        score = 0.0
        txt = str(text)
        for word in positive_words:
            if word in txt:
                score += 1.0
        for word in negative_words:
            if word in txt:
                score -= 1.0
        return score

    @staticmethod
    def _label_sentiment(score: float) -> str:
        if score > 0:
            return "偏多"
        if score < 0:
            return "偏空"
        return "中性"

    @staticmethod
    def _build_news_keywords(stock_name: str, code6: str) -> list[str]:
        name = (stock_name or "").strip()
        aliases = {name, code6}
        cleaned = (
            name.replace("股份有限公司", "")
            .replace("股份", "")
            .replace("集团", "")
            .replace("有限", "")
            .strip()
        )
        if cleaned:
            aliases.add(cleaned)
        if len(cleaned) >= 2:
            aliases.add(cleaned[:2])
        if len(cleaned) >= 3:
            aliases.add(cleaned[:3])
        return [x for x in aliases if x]


def build_data_source(
    data_source_mode: str,
    tushare_token: str,
    tushare_base_url: str | None = None,
) -> StockDataSource:
    mode = (data_source_mode or "auto").strip().lower()
    if mode == "tushare":
        return TushareDataSource(token=tushare_token, base_url=tushare_base_url)
    if mode == "akshare":
        return AkShareDataSource()
    if mode == "auto":
        if (tushare_token or "").strip():
            try:
                return TushareDataSource(token=tushare_token, base_url=tushare_base_url)
            except Exception:
                pass
        return AkShareDataSource()
    raise ValueError("data_source_mode 仅支持 auto/tushare/akshare。")
