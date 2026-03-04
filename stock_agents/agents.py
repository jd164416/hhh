from __future__ import annotations

from typing import List

import pandas as pd

from .models import AgentReport, DecisionResult
from .utils import clamp


class MarketAgent:
    name = "趋势动量 Agent"

    def evaluate(self, df: pd.DataFrame) -> AgentReport:
        last = df.iloc[-1]
        score = 50.0

        if last["close"] > last["ma20"]:
            score += 15
        else:
            score -= 15

        if last["ma20"] > last["ma60"]:
            score += 10
        else:
            score -= 10

        if last["macd_hist"] > 0:
            score += 10
        else:
            score -= 10

        if 45 <= last["rsi14"] <= 70:
            score += 8
        elif last["rsi14"] > 80:
            score -= 8

        score = clamp(score, 0, 100)
        summary = (
            f"收盘 {last['close']:.2f}，MA20 {last['ma20']:.2f}，"
            f"RSI {last['rsi14']:.1f}，MACD柱 {last['macd_hist']:.3f}"
        )
        return AgentReport(
            name=self.name,
            score=score,
            summary=summary,
            details={
                "close_vs_ma20": "强势" if last["close"] > last["ma20"] else "弱势",
                "macd": "多头" if last["macd_hist"] > 0 else "空头",
            },
        )


class FundamentalAgent:
    name = "估值体质 Agent"

    def evaluate(self, merged_df: pd.DataFrame) -> AgentReport:
        last = merged_df.iloc[-1]
        pe = float(last.get("pe_ttm", float("nan")))
        pb = float(last.get("pb", float("nan")))
        turnover = float(last.get("turnover_rate", float("nan")))

        score = 50.0

        if pd.notna(pe):
            if 0 < pe <= 20:
                score += 15
            elif pe <= 35:
                score += 6
            else:
                score -= 10

        if pd.notna(pb):
            if pb <= 2:
                score += 12
            elif pb <= 4:
                score += 5
            else:
                score -= 8

        if pd.notna(turnover):
            if 1 <= turnover <= 8:
                score += 5
            elif turnover > 15:
                score -= 5

        score = clamp(score, 0, 100)
        summary = f"PE(TTM)={pe:.2f} PB={pb:.2f} 换手率={turnover:.2f}%"
        return AgentReport(
            name=self.name,
            score=score,
            summary=summary,
            details={"pe_ttm": f"{pe:.2f}", "pb": f"{pb:.2f}"},
        )


class FlowAgent:
    name = "资金行为 Agent"

    def evaluate(self, merged_df: pd.DataFrame) -> AgentReport:
        last = merged_df.iloc[-1]
        score = 50.0

        net_amount = _first_available(last, ["net_mf_amount", "net_mf_vol", "buy_lg_amount"])
        pct_chg = float(last.get("pct_chg", 0.0))

        if pd.notna(net_amount):
            if net_amount > 0:
                score += 15
            else:
                score -= 15

        if pct_chg > 2:
            score += 6
        elif pct_chg < -2:
            score -= 6

        score = clamp(score, 0, 100)
        summary = f"涨跌幅={pct_chg:.2f}% 资金净额指标={net_amount if pd.notna(net_amount) else '缺失'}"
        return AgentReport(
            name=self.name,
            score=score,
            summary=summary,
            details={"pct_chg": f"{pct_chg:.2f}"},
        )


class RiskAgent:
    name = "风控 Agent"

    def evaluate(self, df: pd.DataFrame) -> AgentReport:
        last = df.iloc[-1]
        atr_pct = float(last["atr14"] / max(last["close"], 1e-9) * 100) if pd.notna(last["atr14"]) else 5.0
        vol20 = float(last["volatility20"]) if pd.notna(last["volatility20"]) else 5.0

        risk_pressure = atr_pct * 3 + vol20 * 2
        score = clamp(100 - risk_pressure * 2, 0, 100)

        if score >= 70:
            level = "低"
        elif score >= 45:
            level = "中"
        else:
            level = "高"

        summary = f"ATR占比={atr_pct:.2f}% 20日波动率={vol20:.2f}% 风险等级={level}"
        return AgentReport(
            name=self.name,
            score=score,
            summary=summary,
            details={"risk_level": level},
        )


class DecisionAgent:
    name = "决策 Agent"

    def combine(self, reports: List[AgentReport], latest_close: float, atr14: float) -> DecisionResult:
        by_name = {r.name: r for r in reports}
        market = by_name["趋势动量 Agent"].score
        fundamental = by_name["估值体质 Agent"].score
        flow = by_name["资金行为 Agent"].score
        risk = by_name["风控 Agent"].score

        overall = 0.40 * market + 0.20 * fundamental + 0.15 * flow + 0.25 * risk

        if overall >= 68:
            action = "BUY"
        elif overall >= 48:
            action = "HOLD"
        else:
            action = "SELL"

        confidence = clamp(45 + abs(overall - 55) * 1.4, 45, 92)
        suggested_position_pct = clamp(20 + risk * 0.6, 20, 80)

        atr = atr14 if pd.notna(atr14) else latest_close * 0.03
        stop_loss = latest_close - 1.5 * atr
        take_profit = latest_close + 2.5 * atr

        if action == "SELL":
            stop_loss = latest_close + 1.5 * atr
            take_profit = latest_close - 2.5 * atr

        summary = (
            f"综合评分 {overall:.1f}。"
            f"趋势={market:.1f} 估值={fundamental:.1f} 资金={flow:.1f} 风险={risk:.1f}。"
        )

        return DecisionResult(
            action=action,
            confidence=confidence,
            suggested_position_pct=suggested_position_pct,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            overall_score=float(overall),
            summary=summary,
            reports=reports,
        )


def _first_available(row: pd.Series, columns: list[str]) -> float:
    for col in columns:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return float("nan")

