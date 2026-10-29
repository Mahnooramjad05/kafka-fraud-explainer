"""Scorer and explainer services over the in-memory bus."""

import json

from conftest import make_tx

from fraud_explainer.explainer import ATTEMPT_HEADER, RETRY_AT_HEADER, ExplainerService
from fraud_explainer.llm import ExplainerError, MockExplainer
from fraud_explainer.schemas import Alert, DeadLetter, Explanation, ScoredTransaction
from fraud_explainer.scorer import ScorerService


async def send(bus, settings, txs):
    producer = await bus.producer()
    for tx in txs:
        await producer.send(settings.topics.raw, tx.model_dump_json().encode(), key=tx.account_id)


def normal_history(account="acct-1", n=10):
    return [make_tx(i, account=account, minutes=-600 + i * 60) for i in range(n)]


def card_testing(account="acct-1", start=100, n=6):
    return [make_tx(start + i, account=account, minutes=i * 0.5, amount=1.5, category="online_retail")
            for i in range(n)]


async def drain(service):
    while await service.run_once(timeout_ms=1):
        pass


async def run_scorer(bus, settings, model, metrics):
    scorer = ScorerService(bus, settings, model, metrics)
    await scorer.start()
    await drain(scorer)
    return scorer


def scored(bus, settings):
    return [ScoredTransaction.model_validate_json(r.value) for r in bus.all(settings.topics.scored)]


async def test_scorer_flags_card_testing_and_keeps_state(bus, settings, model, metrics):
    await send(bus, settings, normal_history() + card_testing())
    await run_scorer(bus, settings, model, metrics)

    results = {s.transaction.transaction_id: s for s in scored(bus, settings)}
    assert len(results) == 16
    assert all(results[f"tx-acct-1-{i}"].risk < settings.risk_threshold for i in range(10))
    late = results["tx-acct-1-105"]
    assert late.risk >= settings.risk_threshold
    assert "card_testing" in [s.name for s in late.signals]

    snapshot = bus.latest_by_key(settings.topics.account_state)["acct-1"]
    assert json.loads(snapshot.value)["n"] == 13  # 10 normal + 3 tests before the rule fired
    assert metrics.get("transactions_flagged_total") >= 3


async def test_malformed_messages_go_to_dead_letter_queue(bus, settings, model, metrics):
    producer = await bus.producer()
    await producer.send(settings.topics.raw, b"not json", key="acct-9")
    await producer.send(settings.topics.raw, b'{"transaction_id": "x", "amount": -5}', key="acct-9")
    await send(bus, settings, [make_tx(1)])
    await run_scorer(bus, settings, model, metrics)

    letters = [DeadLetter.model_validate_json(r.value) for r in bus.all(settings.topics.dlq)]
    assert len(letters) == 2 and {d.stage for d in letters} == {"scorer"}
    assert len(scored(bus, settings)) == 1
    assert metrics.get("dead_letters_total", stage="scorer") == 2


async def test_duplicates_are_dropped(bus, settings, model, metrics):
    txs = normal_history()
    await send(bus, settings, txs + [txs[3], txs[7]])
    await run_scorer(bus, settings, model, metrics)
    assert len(scored(bus, settings)) == 10
    assert metrics.get("scorer_duplicates_total") == 2


async def test_new_scorer_restores_state_and_dedupe_from_compacted_topic(bus, settings, model, metrics):
    """Simulates a crash/rebalance: a new instance picks up the partitions."""
    await send(bus, settings, normal_history())
    await run_scorer(bus, settings, model, metrics)

    # Replay an already-processed transaction plus new card-testing traffic.
    await send(bus, settings, [normal_history()[0]] + card_testing())
    fresh = await run_scorer(bus, settings, model, metrics)

    assert fresh.states["acct-1"].n >= 10            # history came back from account-state
    assert metrics.get("scorer_duplicates_total") == 1  # the replay was recognized
    flagged = [s for s in scored(bus, settings) if s.risk >= settings.risk_threshold]
    assert flagged, "card testing should be caught using restored history"


async def test_committed_offsets_prevent_reprocessing(bus, settings, model, metrics):
    await send(bus, settings, normal_history())
    await run_scorer(bus, settings, model, metrics)
    await run_scorer(bus, settings, model, metrics)
    assert len(scored(bus, settings)) == 10


class CountingExplainer(MockExplainer):
    def __init__(self, fail_times: int = 0, hallucinate: bool = False):
        self.calls, self.fail_times, self.hallucinate = [], fail_times, hallucinate

    async def explain(self, batch):
        self.calls.append(len(batch))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ExplainerError("provider timeout")
        out = await super().explain(batch)
        if self.hallucinate:
            out = [Explanation(**{**e.model_dump(), "cited_signals": e.cited_signals + ["vpn_detected"]})
                   for e in out]
        return out


async def pipeline(bus, settings, model, metrics, explainer, txs):
    await send(bus, settings, txs)
    await run_scorer(bus, settings, model, metrics)
    service = ExplainerService(bus, settings, explainer, metrics, clock=lambda: 0.0, sleep=_no_sleep)
    await service.start()
    await drain(service)
    return [Alert.model_validate_json(r.value) for r in bus.all(settings.topics.alerts)]


async def _no_sleep(_seconds):
    return None


async def test_only_flagged_transactions_reach_the_llm_in_batches(bus, settings, model, metrics):
    explainer = CountingExplainer()
    accounts = [f"acct-{i}" for i in range(3)]
    txs = [tx for a in accounts for tx in normal_history(a) + card_testing(a)]
    alerts = await pipeline(bus, settings, model, metrics, explainer, txs)

    flagged = [s for s in scored(bus, settings) if s.risk >= settings.risk_threshold]
    assert len(alerts) == len(flagged) == sum(explainer.calls)
    assert max(explainer.calls) <= settings.batch_size
    assert all(a.faithful for a in alerts)
    assert all(a.explanation.cited_signals for a in alerts)


async def test_failed_batch_is_retried_then_succeeds(bus, settings, model, metrics):
    explainer = CountingExplainer(fail_times=1)
    alerts = await pipeline(bus, settings, model, metrics, explainer, normal_history() + card_testing())
    retries = bus.all(settings.topics.retry)
    assert retries and all(r.headers[ATTEMPT_HEADER] == "2" for r in retries)
    assert RETRY_AT_HEADER in retries[0].headers
    assert len(alerts) == len(retries)  # every retried event was eventually explained
    assert not bus.all(settings.topics.dlq)


async def test_persistent_failure_ends_in_dead_letter_queue(bus, settings, model, metrics):
    explainer = CountingExplainer(fail_times=100)
    alerts = await pipeline(bus, settings, model, metrics, explainer, normal_history() + card_testing())
    letters = [DeadLetter.model_validate_json(r.value) for r in bus.all(settings.topics.dlq)]
    assert not alerts
    assert letters and all(d.stage == "explainer" and d.attempts == settings.max_attempts for d in letters)
    assert metrics.get("llm_batches_total", status="error") >= settings.max_attempts


async def test_unfaithful_explanations_are_flagged(bus, settings, model, metrics):
    alerts = await pipeline(bus, settings, model, metrics, CountingExplainer(hallucinate=True),
                            normal_history() + card_testing())
    assert alerts and not any(a.faithful for a in alerts)
    assert all(a.unsupported_signals == ["vpn_detected"] for a in alerts)
    assert metrics.get("unfaithful_explanations_total") == len(alerts)
