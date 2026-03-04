from __future__ import annotations

from datetime import date, datetime


def normalize_cn_code(raw_code: str) -> str:
    code = raw_code.strip().upper()
    if "." in code:
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        return f"{code}.SZ"
    raise ValueError("股票代码格式错误，请输入 6 位代码，例如 600519 或 000001")


def to_tushare_date(d: date | datetime | str) -> str:
    if isinstance(d, str):
        dt = datetime.strptime(d, "%Y-%m-%d")
    elif isinstance(d, datetime):
        dt = d
    else:
        dt = datetime.combine(d, datetime.min.time())
    return dt.strftime("%Y%m%d")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

