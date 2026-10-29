from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from conftest import make_tx
from fastapi.testclient import TestClient

from eval.detection import evaluate
from fraud_explainer.api import AlertStore, create_app
from fraud_explainer.llm import (
    AnthropicExplainer,
    ExplainerError,
    MockExplainer,
    OpenAIExplainer,
    build_explainer,
    check_faithfulness,
    parse_explanations,
)
from fraud_explainer.metrics import Metrics
from fraud_explainer.schemas import Alert, Explanation, Features, ScoredTransaction, Signal

NOW = datetime(2026, 3, 2, tzinfo=timezone.utc)


def flagged(i=1, risk=0.9, signals=("card_testing",)) -> ScoredTransaction:
    return ScoredTransaction(
        transaction=make_tx(i), features=Features(), model_score=0.5, risk=risk, scored_at=NOW,
        signals=[Signal(name=n, detail=f"{n} detail", weight=0.8) for n in signals],
    )


def test_faithfulness_check():
    s = flagged(signals=("card_testing", "anomaly_model"))
    ok = Explanation(transaction_id="t", summary="", cited_signals=["card_testing"],
                     recommended_action="block", confidence=0.9)
    bad = ok.model_copy(update={"cited_signals": ["card_testing", "vpn_detected"]})
    empty = ok.model_copy(update={"cited_signals": []})
    assert check_faithfulness(s, ok) == (True, [])
    assert check_faithfulness(s, bad) == (False, ["vpn_detected"])
    assert check_faithfulness(s, empty) == (False, [])


def test_parse_explanations_requires_every_transaction():
    batch = [flagged(1), flagged(2)]
    one = {"explanations": [{"transaction_id": "tx-acct-1-1", "summary": "s", "cited_signals": [],
                             "recommended_action": "review", "confidence": 0.5}]}
    with pytest.raises(ExplainerError, match="tx-acct-1-2"):
        parse_explanations(one, batch)


async def test_mock_explainer_cites_fired_signals():
    [exp] = await MockExplainer().explain([flagged(risk=0.9)])
    assert exp.cited_signals == ["card_testing"] and exp.recommended_action == "block"


def _explanations_json(batch):
    import json
    return json.dumps({"explanations": [
        {"transaction_id": s.transaction.transaction_id, "summary": "x", "cited_signals": ["card_testing"],
         "recommended_action": "block", "confidence": 0.9} for s in batch]})


async def test_anthropic_request_shape():
    batch = [flagged(1), flagged(2)]
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=_explanations_json(batch))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    explanations = await AnthropicExplainer(client=client).explain(batch)
    assert len(explanations) == 2
    assert captured["model"] == "claude-opus-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["fallbacks"] == "default"
    assert "temperature" not in captured
    assert "tx-acct-1-2" in captured["messages"][0]["content"]


async def test_anthropic_refusal_raises():
    async def create(**kwargs):
        return SimpleNamespace(stop_reason="refusal", content=[])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    with pytest.raises(ExplainerError, match="declined"):
        await AnthropicExplainer(client=client).explain([flagged()])


async def test_openai_request_shape():
    batch = [flagged(1)]
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content=_explanations_json(batch), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    [exp] = await OpenAIExplainer(client=client).explain(batch)
    assert exp.recommended_action == "block"
    assert captured["response_format"]["json_schema"]["strict"] is True


def test_build_explainer_falls_back_to_mock_without_key():
    assert build_explainer("anthropic").name == "mock"
    assert build_explainer("mock").name == "mock"


def make_alert(i, risk, action="review") -> Alert:
    s = flagged(i, risk=risk)
    return Alert(transaction_id=s.transaction.transaction_id, account_id="acct-1", amount=1.0,
                 merchant="m", city="c", risk=risk, signals=s.signals, faithful=True, explainer="mock",
                 created_at=NOW, explanation=Explanation(
                     transaction_id=s.transaction.transaction_id, summary="s",
                     cited_signals=["card_testing"], recommended_action=action, confidence=risk))


def test_alert_store_order_filters_and_capacity():
    store = AlertStore(capacity=3)
    for i, (risk, action) in enumerate([(0.7, "review"), (0.9, "block"), (0.95, "block"), (0.65, "review")]):
        store.add(make_alert(i, risk, action))
    assert len(store) == 3                                     # oldest evicted
    assert [a.risk for a in store.recent(10)] == [0.65, 0.95, 0.9]
    assert [a.risk for a in store.recent(10, min_risk=0.9)] == [0.95, 0.9]
    assert [a.risk for a in store.recent(10, action="review")] == [0.65]


def test_api_endpoints():
    store, metrics = AlertStore(), Metrics()
    store.add(make_alert(1, 0.9, "block"))
    metrics.inc("alerts_total", action="block", help="Alerts produced")
    metrics.observe("llm_batch_seconds", 0.3)
    client = TestClient(create_app(store=store, metrics=metrics))

    assert client.get("/health").json()["alerts_loaded"] == 1
    assert client.get("/alerts?action=block").json()[0]["transaction_id"] == "tx-acct-1-1"
    assert client.get("/alerts/tx-acct-1-1").status_code == 200
    assert client.get("/alerts/missing").status_code == 404
    assert client.get("/alerts?action=delete").status_code == 422
    text = client.get("/metrics").text
    assert 'alerts_total{action="block"} 1' in text
    assert 'llm_batch_seconds_bucket{le="0.5"} 1' in text


async def test_end_to_end_eval_on_small_stream():
    result = await evaluate(num_events=3000, fraud_rate=0.04, seed=11, train_events=2000)
    assert result.events == 3000
    assert result.precision >= 0.8
    assert result.incident_recall >= 0.9
    assert result.llm_share < 0.15
    assert result.alerts == result.flagged and result.faithful_rate == 1.0
