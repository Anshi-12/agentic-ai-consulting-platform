"""D. Financial & Scenario Agent.

Numbers come exclusively from `request_numerical_analysis` (deterministic
Python in backend/ai/finance.py). The LLM only interprets them, and its text is
checked for figures that do not match any calculated value.
"""
from __future__ import annotations

import re

from backend.agents.base import AgentContext, AgentResult, BaseAgent
from backend.ai.structured_outputs import FinancialInterpretation, FinancialOutput, ScenarioMetrics
from backend.prompts.financial_prompts import FINANCIAL_PROMPT

_MONEY = re.compile(r"\$\s?(-?\d+(?:[.,]\d+)*)\s*([kKmMbB]n?|million|billion|thousand)?")


def parse_money(text: str) -> list[float]:
    out = []
    for num, unit in _MONEY.findall(text or ""):
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        if u.startswith("k") or u == "thousand":
            val *= 1e3
        elif u.startswith("m"):
            val *= 1e6
        elif u.startswith("b"):
            val *= 1e9
        out.append(val)
    return out


def unverified_amounts(texts: list[str], known: list[float], tolerance: float = 0.06) -> list[float]:
    """Currency amounts in `texts` not within `tolerance` of any known calculated value."""
    bad = []
    for amount in (a for t in texts for a in parse_money(t)):
        if not any(abs(amount - k) <= tolerance * max(abs(k), 1.0) for k in known if k is not None):
            bad.append(amount)
    return bad


class FinancialScenarioAgent(BaseAgent):
    name = "financial_scenario"

    def run(self, ctx: AgentContext) -> AgentResult:
        result = AgentResult(self.name, {})
        session = self.session(ctx)
        res = session.call("request_numerical_analysis", project_id=ctx.project_id, analysis="scenario_metrics")

        scenarios: list[ScenarioMetrics] = []
        calc_log: list[str] = []
        currency = "USD"
        if res.success:
            scenarios = [ScenarioMetrics.model_validate(s) for s in res.data["scenarios"]]
            calc_log = res.data["calculation_log"]
            currency = res.data.get("currency", "USD")
        data_available = bool(res.success and scenarios)

        trusted = {
            "data_available": data_available,
            "currency": currency,
            "scenarios": [s.model_dump(mode="json") for s in scenarios],
            "tool_error": None if res.success else res.error,
        }
        interp: FinancialInterpretation = self.llm(FINANCIAL_PROMPT, trusted, tool_log=session.execution_log(),
                                                   result=result)

        # Deterministic post-checks
        names = {s.scenario for s in scenarios}
        if not data_available and interp.data_available:
            result.warnings.append("model claimed data_available without successful calculation; corrected")
            interp.data_available = False
            interp.preferred_scenario = None
        if interp.preferred_scenario and interp.preferred_scenario not in names:
            result.warnings.append(f"preferred_scenario '{interp.preferred_scenario}' not a supplied scenario; cleared")
            interp.preferred_scenario = None
        known = [m.value for s in scenarios for m in s.metrics if m.value is not None and m.unit == "currency"]
        bad = unverified_amounts([interp.summary, *interp.scenario_comparison, *interp.key_drivers], known)
        if bad:
            interp.caveats.append(f"Unverified figures in narrative (not produced by calculation): {bad[:5]}")
            interp.confidence = min(interp.confidence, 0.5)
            result.warnings.append(f"unverified currency amounts in interpretation: {bad[:5]}")

        out = FinancialOutput(data_available=data_available, currency=currency, scenarios=scenarios,
                              interpretation=interp, calculation_log=calc_log)
        result.output = {"financial": out.model_dump(mode="json")}
        return self.finish(result, session)
