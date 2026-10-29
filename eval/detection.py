"""
Evaluate the pipeline end to end on labeled simulated traffic.

Runs the real scorer and explainer services over an in-memory bus (no broker
needed), then compares alerts against the simulator's ground truth:

  - transaction level: precision / recall / F1 of "risk >= threshold"
  - incident level   : share of fraud incidents with at least one flagged event
    (a card-testing burst only becomes detectable after a few charges)
  - per pattern      : recall for each fraud pattern
  - LLM              : share of traffic sent to the LLM, and how many
                       explanations were faithful to the signals that fired

Usage:
    python -m eval.detection [--events 10000] [--fraud-rate 0.03] [--seed 42] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from fraud_explainer.bus import InMemoryBus
from fraud_explainer.config import Settings
from fraud_explainer.explainer import ExplainerService
from fraud_explainer.llm import Explainer, MockExplainer
from fraud_explainer.metrics import Metrics
from fraud_explainer.model import DetectionModel
from fraud_explainer.producer import publish
from fraud_explainer.schemas import Alert, ScoredTransaction
from fraud_explainer.scorer import ScorerService
from fraud_explainer.simulator import PATTERNS, Simulator


@dataclass
class EvalResult:
    events: int
    fraud_events: int
    flagged: int
    precision: float
    recall: float
    f1: float
    incident_recall: float
    pattern_recall: dict[str, float]
    pattern_incident_recall: dict[str, float]
    false_positive_signals: dict[str, int]
    llm_share: float
    alerts: int
    faithful_rate: float
    actions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


async def drain(service, idle_polls: int = 3) -> None:
    idle = 0
    while idle < idle_polls:
        idle = idle + 1 if await service.run_once(timeout_ms=1) == 0 else 0


async def evaluate(num_events: int = 10_000, fraud_rate: float = 0.03, seed: int = 42,
                   train_events: int = 6000, explainer: Explainer | None = None,
                   settings: Settings | None = None) -> EvalResult:
    settings = settings or Settings()
    bus = InMemoryBus(settings)
    await bus.ensure_topics()
    metrics = Metrics()

    model = DetectionModel().fit(Simulator(seed=seed + 1000, fraud_rate=0).normal_history(train_events))
    sent = await publish(bus, settings, Simulator(seed=seed, fraud_rate=fraud_rate), num_events)

    scorer = ScorerService(bus, settings, model, metrics)
    await scorer.start()
    await drain(scorer)
    explain = ExplainerService(bus, settings, explainer or MockExplainer(), metrics)
    await explain.start()
    await drain(explain)

    scored = {s.transaction.transaction_id: s for s in
              (ScoredTransaction.model_validate_json(r.value) for r in bus.all(settings.topics.scored))}
    alerts = [Alert.model_validate_json(r.value) for r in bus.all(settings.topics.alerts)]

    tp = fp = fn = 0
    by_pattern, flagged_by_pattern = Counter(), Counter()
    incidents: dict[str, bool] = {}
    incident_pattern: dict[str, str] = {}
    fp_signals: Counter = Counter()
    for event in sent:
        s = scored[event.transaction.transaction_id]
        flagged = s.risk >= settings.risk_threshold
        if event.label:
            by_pattern[event.label] += 1
            flagged_by_pattern[event.label] += flagged
            incidents[event.incident_id] = incidents.get(event.incident_id, False) or flagged
            incident_pattern[event.incident_id] = event.label
            tp += flagged
            fn += not flagged
        elif flagged:
            fp += 1
            fp_signals.update(sig.name for sig in s.signals)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    inc_by_pattern: dict[str, list[bool]] = defaultdict(list)
    for incident, hit in incidents.items():
        inc_by_pattern[incident_pattern[incident]].append(hit)

    return EvalResult(
        events=len(sent),
        fraud_events=tp + fn,
        flagged=tp + fp,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        incident_recall=round(sum(incidents.values()) / len(incidents), 4) if incidents else 0.0,
        pattern_recall={p: round(flagged_by_pattern[p] / by_pattern[p], 4) for p in PATTERNS if by_pattern[p]},
        pattern_incident_recall={p: round(sum(v) / len(v), 4) for p, v in sorted(inc_by_pattern.items())},
        false_positive_signals=dict(fp_signals.most_common()),
        llm_share=round((tp + fp) / len(sent), 4),
        alerts=len(alerts),
        faithful_rate=round(sum(a.faithful for a in alerts) / len(alerts), 4) if alerts else 0.0,
        actions=dict(Counter(a.explanation.recommended_action for a in alerts)),
    )


def print_result(r: EvalResult) -> None:
    print("=" * 64)
    print("FRAUD DETECTION EVAL (simulated, labeled traffic)")
    print("=" * 64)
    print(f"Events: {r.events}   fraud events: {r.fraud_events}   flagged: {r.flagged}")
    print(f"Transaction level  precision {r.precision:.3f}  recall {r.recall:.3f}  F1 {r.f1:.3f}")
    print(f"Incident recall    {r.incident_recall:.3f}  (incident caught if any of its events is flagged)")
    print("\nPer pattern              event recall   incident recall")
    for p in PATTERNS:
        if p in r.pattern_recall:
            print(f"  {p:<22} {r.pattern_recall[p]:>10.3f} {r.pattern_incident_recall.get(p, 0):>16.3f}")
    print(f"\nFalse-positive signals: {r.false_positive_signals or 'none'}")
    print(f"Sent to LLM: {r.llm_share:.1%} of traffic   alerts: {r.alerts}   "
          f"faithful explanations: {r.faithful_rate:.1%}   actions: {r.actions}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--fraud-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.events, args.fraud_rate, args.seed))
    print_result(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)


if __name__ == "__main__":
    main()
