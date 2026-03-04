from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

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
