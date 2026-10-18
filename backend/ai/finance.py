"""Deterministic financial calculations.

LLMs never do arithmetic in this system. The Financial & Scenario agent calls
these functions (via the `request_numerical_analysis` tool) and only
*interprets* the results. Each metric carries a `value_type` so calculated
values are distinguishable from the estimates/forecasts they are based on.
"""
from __future__ import annotations

from typing import Any

from backend.ai.structured_outputs import ClaimType, MetricValue, ScenarioMetrics


class FinanceInputError(ValueError):
    pass


def roi(total_return: float, investment: float) -> float | None:
    """Simple ROI = (total cash returned - investment) / investment."""
    if investment <= 0:
        return None
    return (total_return - investment) / investment


def payback_period(investment: float, annual_cash_flows: list[float]) -> float | None:
    """Years until cumulative cash flow recovers the investment (linear interpolation within a year)."""
    if investment <= 0:
        return 0.0
    cumulative = 0.0
    for year, cf in enumerate(annual_cash_flows, start=1):
        prev = cumulative
        cumulative += cf
        if cumulative >= investment:
            if cf <= 0:
                return float(year)
            return (year - 1) + (investment - prev) / cf
    return None  # not recovered within horizon


def npv(rate: float, investment: float, annual_cash_flows: list[float]) -> float:
    if rate <= -1:
        raise FinanceInputError("discount rate must be > -100%")
    return -investment + sum(cf / (1 + rate) ** t for t, cf in enumerate(annual_cash_flows, start=1))


def irr(investment: float, annual_cash_flows: list[float], lo: float = -0.99, hi: float = 10.0,
        tol: float = 1e-7, max_iter: int = 200) -> float | None:
    """IRR via bisection; None if no sign change in the bracket."""
    f = lambda r: npv(r, investment, annual_cash_flows)  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def gross_margin(revenue: float, cogs: float) -> float | None:
    return None if revenue <= 0 else (revenue - cogs) / revenue


def operating_margin(revenue: float, cogs: float, opex: float) -> float | None:
    return None if revenue <= 0 else (revenue - cogs - opex) / revenue


def _num(d: dict[str, Any], key: str, required: bool = True) -> float | None:
    val = d.get(key)
    if val is None:
        if required:
            raise FinanceInputError(f"missing numeric field '{key}'")
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise FinanceInputError(f"field '{key}' must be numeric")
    return float(val)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def compute_scenarios(financials: dict[str, Any]) -> tuple[list[ScenarioMetrics], list[str]]:
    """Compute all metrics for every scenario. Returns (scenario metrics, calculation log)."""
    if not financials:
        raise FinanceInputError("no financial inputs supplied")
    rate = _num(financials, "discount_rate", required=False)
    rate = 0.10 if rate is None else rate
    log = [f"discount_rate={rate} ({'supplied' if 'discount_rate' in financials else 'default'})"]
    results: list[ScenarioMetrics] = []

    baseline = financials.get("baseline")
    if baseline:
        rev, cogs, opex = _num(baseline, "revenue"), _num(baseline, "cogs"), _num(baseline, "opex", False)
        metrics = [MetricValue(name="gross_margin", value=_r(gross_margin(rev, cogs)), unit="ratio",
                               value_type=ClaimType.CALCULATED, note="from supplied baseline actuals")]
        if opex is not None:
            metrics.append(MetricValue(name="operating_margin", value=_r(operating_margin(rev, cogs, opex)),
                                       unit="ratio", value_type=ClaimType.CALCULATED))
        results.append(ScenarioMetrics(scenario="Baseline", metrics=metrics, input_value_type=ClaimType.FACT))
        log.append("baseline: gross_margin=(revenue-cogs)/revenue")

    for sc in financials.get("scenarios", []):
        name = str(sc.get("name", "Unnamed scenario"))[:120]
        inv = _num(sc, "investment")
        cfs = sc.get("annual_cash_flows") or []
        if not isinstance(cfs, list) or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in cfs):
            raise FinanceInputError(f"scenario '{name}': annual_cash_flows must be a list of numbers")
        cfs = [float(c) for c in cfs]
        total = sum(cfs)
        try:
            input_type = ClaimType(sc.get("value_type", "estimate"))
        except ValueError:
            input_type = ClaimType.ESTIMATE
        m = [
            MetricValue(name="investment", value=inv, unit="currency", value_type=input_type, source="data_module",
                        note="supplied input"),
            MetricValue(name="total_cash_flow", value=_r(total, 2), unit="currency", value_type=ClaimType.CALCULATED,
                        note=f"sum over {len(cfs)} years of {input_type.value} cash flows"),
            MetricValue(name="roi", value=_r(roi(total, inv)), unit="ratio", value_type=ClaimType.CALCULATED),
            MetricValue(name="payback_years", value=_r(payback_period(inv, cfs), 2), unit="years",
                        value_type=ClaimType.CALCULATED,
                        note="" if payback_period(inv, cfs) is not None else "not recovered within horizon"),
            MetricValue(name="npv", value=_r(npv(rate, inv, cfs), 2), unit="currency", value_type=ClaimType.CALCULATED,
                        note=f"at {rate:.1%} discount rate"),
            MetricValue(name="irr", value=_r(irr(inv, cfs)), unit="ratio", value_type=ClaimType.CALCULATED),
        ]
        if "revenue" in sc and "cogs" in sc:
            rev, cogs = _num(sc, "revenue"), _num(sc, "cogs")
            m.append(MetricValue(name="gross_margin", value=_r(gross_margin(rev, cogs)), unit="ratio",
                                 value_type=ClaimType.CALCULATED, note=f"on {input_type.value} revenue/cogs"))
            if "opex" in sc:
                m.append(MetricValue(name="operating_margin", value=_r(operating_margin(rev, cogs, _num(sc, "opex"))),
                                     unit="ratio", value_type=ClaimType.CALCULATED))
        results.append(ScenarioMetrics(scenario=name, metrics=m, input_value_type=input_type,
                                       assumptions=[str(a)[:300] for a in sc.get("assumptions", [])]))
        log.append(f"{name}: roi, payback, npv, irr, margins computed from {len(cfs)} cash-flow years")
    return results, log


def metric_lookup(scenarios: list[ScenarioMetrics]) -> dict[tuple[str, str], float | None]:
    return {(s.scenario, m.name): m.value for s in scenarios for m in s.metrics}
