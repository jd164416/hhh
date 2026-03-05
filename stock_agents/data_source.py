from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re

import pandas as pd
import tushare as ts
import tushare.pro.client as ts_client

from .utils import to_tushare_date


@dataclass
class MarketDataBundle:
    daily: pd.DataFrame
    daily_basic: pd.DataFrame
    moneyflow: pd.DataFrame


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

        try:
            news = self.pro.news(
                start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception:
            return pd.DataFrame(
                columns=["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
            )

        if news is None or news.empty:
            return pd.DataFrame(
                columns=["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
            )

        df = news.copy()
        for col in ["datetime", "title", "content", "src"]:
            if col not in df.columns:
                df[col] = ""

        text = (
            df["title"].astype(str).fillna("")
            + " "
            + df["content"].astype(str).fillna("")
        ).str.lower()
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
            fallback_text = (
                fallback["title"].astype(str).fillna("")
                + " "
                + fallback["content"].astype(str).fillna("")
            ).str.lower()
            fallback["sentiment_score"] = fallback_text.apply(self._score_text_sentiment).astype(float)
            fallback["sentiment_label"] = fallback["sentiment_score"].apply(self._label_sentiment)
            fallback["source"] = fallback["src"].astype(str)
            fallback["match_type"] = "市场新闻(未精准匹配)"
            keep_cols = ["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
            return fallback[keep_cols].reset_index(drop=True)

        df["sentiment_score"] = (
            text.loc[df.index].apply(self._score_text_sentiment).astype(float)
        )
        df["sentiment_label"] = df["sentiment_score"].apply(self._label_sentiment)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime", ascending=False).head(limit)
        df["source"] = df["src"].astype(str)
        df["match_type"] = "个股匹配"

        keep_cols = ["datetime", "title", "source", "sentiment_score", "sentiment_label", "match_type"]
        return df[keep_cols].reset_index(drop=True)

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
    def _score_text_sentiment(text: str) -> float:
        positive_words = [
            "增长",
            "利好",
            "突破",
            "上调",
            "回购",
            "超预期",
            "盈利",
            "签约",
            "中标",
            "看多",
        ]
        negative_words = [
            "下滑",
            "利空",
            "暴跌",
            "下调",
            "亏损",
            "减持",
            "违约",
            "处罚",
            "看空",
            "风险",
        ]

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
