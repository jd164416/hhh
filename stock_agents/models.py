from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentReport:
    name: str
    score: float
    summary: str
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class DecisionResult:
    action: str
    confidence: float
    suggested_position_pct: float
    stop_loss: float
    take_profit: float
    overall_score: float
    summary: str
    reports: List[AgentReport]

