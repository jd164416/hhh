from __future__ import annotations

from typing import Optional

from openai import OpenAI

from .models import DecisionResult


def _build_prompt(ticker: str, trade_date: str, decision: DecisionResult) -> str:
    report_text = "\n".join([f"- {r.name}: 评分 {r.score:.1f}，{r.summary}" for r in decision.reports])
    return (
        f"股票: {ticker}\n"
        f"日期: {trade_date}\n"
        f"系统动作: {decision.action}\n"
        f"综合评分: {decision.overall_score:.1f}\n"
        f"置信度: {decision.confidence:.1f}\n"
        f"建议仓位: {decision.suggested_position_pct:.0f}%\n"
        f"止损位: {decision.stop_loss:.2f}\n"
        f"止盈位: {decision.take_profit:.2f}\n"
        f"各Agent结果:\n{report_text}\n\n"
        "请输出中文，结构固定为：\n"
        "1) 一句话结论\n"
        "2) 三条核心依据\n"
        "3) 风险提示（两条）\n"
        "4) 明日执行清单（两条，必须可执行）\n"
        "不要给空泛内容，不要承诺收益。"
    )


class DeepSeekExplainer:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=45, max_retries=2)

    def explain(self, ticker: str, trade_date: str, decision: DecisionResult) -> str:
        prompt = _build_prompt(ticker, trade_date, decision)
        resp = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是谨慎、可执行、面向散户的A股交易助理。"},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class GeminiExplainer:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def explain(self, ticker: str, trade_date: str, decision: DecisionResult) -> str:
        prompt = _build_prompt(ticker, trade_date, decision)
        from google import genai

        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(
            model=self.model,
            contents=[
                "你是谨慎、可执行、面向散户的A股交易助理。",
                prompt,
            ],
        )
        text = getattr(resp, "text", "") or ""
        return text.strip()


def maybe_explain(
    provider: str,
    deepseek_api_key: Optional[str],
    gemini_api_key: Optional[str],
    ticker: str,
    trade_date: str,
    decision: DecisionResult,
) -> str:
    provider_lower = (provider or "deepseek").strip().lower()

    try:
        if provider_lower == "gemini":
            if not gemini_api_key:
                return "未配置 Gemini API Key，当前展示规则引擎结论。"
            return GeminiExplainer(api_key=gemini_api_key).explain(ticker, trade_date, decision)

        if not deepseek_api_key:
            return "未配置 DeepSeek API Key，当前展示规则引擎结论。"
        return DeepSeekExplainer(api_key=deepseek_api_key).explain(ticker, trade_date, decision)
    except Exception as exc:
        provider_name = "Gemini" if provider_lower == "gemini" else "DeepSeek"
        return f"{provider_name} 解释生成失败，已保留规则引擎结论。错误: {exc}"
