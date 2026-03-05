from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .agents import DecisionAgent, FlowAgent, FundamentalAgent, MarketAgent, RiskAgent
from .data_source import AkShareDataSource, build_data_source
from .indicators import add_indicators
from .models import DecisionResult


@dataclass
class AnalysisOutput:
    enriched_df: pd.DataFrame
    merged_df: pd.DataFrame
    decision: DecisionResult
    used_trade_date: str


class RetailStockAgentsEngine:
    def __init__(
        self,
        tushare_token: str,
        tushare_base_url: str | None = None,
        data_source_mode: str = "auto",
    ):
        self.data_source_mode = (data_source_mode or "auto").strip().lower()
        self.data_source = build_data_source(
            data_source_mode=self.data_source_mode,
            tushare_token=tushare_token,
            tushare_base_url=tushare_base_url,
        )
        self.market_agent = MarketAgent()
        self.fundamental_agent = FundamentalAgent()
        self.flow_agent = FlowAgent()
        self.risk_agent = RiskAgent()
        self.decision_agent = DecisionAgent()

    def analyze(
        self,
        ts_code: str,
        trade_date: str,
        lookback_days: int = 260,
    ) -> AnalysisOutput:
        try:
            bundle = self.data_source.fetch_bundle(
                ts_code=ts_code,
                analysis_date=trade_date,
                lookback_days=lookback_days,
            )
        except Exception:
            if self.data_source_mode == "auto" and not isinstance(self.data_source, AkShareDataSource):
                self.data_source = AkShareDataSource()
                bundle = self.data_source.fetch_bundle(
                    ts_code=ts_code,
                    analysis_date=trade_date,
                    lookback_days=lookback_days,
                )
            else:
                raise
        enriched = add_indicators(bundle.daily)
        merged = self._merge_features(enriched, bundle.daily_basic, bundle.moneyflow)

        mkt = self.market_agent.evaluate(enriched)
        fnd = self.fundamental_agent.evaluate(merged)
        flw = self.flow_agent.evaluate(merged)
        rsk = self.risk_agent.evaluate(enriched)

        last = enriched.iloc[-1]
        decision = self.decision_agent.combine(
            reports=[mkt, fnd, flw, rsk],
            latest_close=float(last["close"]),
            atr14=float(last["atr14"]) if pd.notna(last["atr14"]) else float("nan"),
        )
        used_trade_date = pd.to_datetime(last["trade_date"]).strftime("%Y-%m-%d")
        return AnalysisOutput(
            enriched_df=enriched,
            merged_df=merged,
            decision=decision,
            used_trade_date=used_trade_date,
        )

    @staticmethod
    def _merge_features(
        daily_df: pd.DataFrame,
        daily_basic_df: pd.DataFrame,
        moneyflow_df: pd.DataFrame,
    ) -> pd.DataFrame:
        out = daily_df.copy()
        if not daily_basic_df.empty:
            keep = [c for c in ["trade_date", "pe_ttm", "pb", "turnover_rate"] if c in daily_basic_df.columns]
            out = out.merge(daily_basic_df[keep], on="trade_date", how="left")
        if not moneyflow_df.empty:
            keep = [c for c in ["trade_date", "net_mf_amount", "net_mf_vol", "buy_lg_amount"] if c in moneyflow_df.columns]
            out = out.merge(moneyflow_df[keep], on="trade_date", how="left")
        return out
