from datetime import timedelta

import pytest
from conftest import make_tx

from fraud_explainer.features import AccountState, haversine_km
from fraud_explainer.model import DetectionModel, combine, rule_signals
from fraud_explainer.schemas import Features, Signal
from fraud_explainer.simulator import PATTERNS, Simulator


def history(state: AccountState, n: int = 10, amount: float = 40.0) -> None:
    for i in range(n):
        state.update(make_tx(i, minutes=-600 + i * 60, amount=amount))


def test_haversine_lahore_to_london():
    assert haversine_km(31.52, 74.36, 51.507, -0.128) == pytest.approx(6290, rel=0.02)


def test_features_use_history_before_the_transaction():
    state = AccountState("acct-1")
    history(state)
    f = state.features_for(make_tx(99, amount=400.0))
    assert f.history_size == 10
    assert f.amount_ratio_to_mean == pytest.approx(10.0)
    assert f.amount_zscore > 4
    assert f.new_category is False


def test_amount_features_need_minimum_history():
    state = AccountState("acct-1")
    history(state, n=3)
    f = state.features_for(make_tx(99, amount=4000.0, category="jewelry"))
    assert f.amount_zscore == 0 and f.new_category is False


def test_small_charge_window_counts_only_last_10_minutes():
    state = AccountState("acct-1")
    history(state)
    for i in range(3):
        state.update(make_tx(100 + i, minutes=i, amount=1.0))
    f = state.features_for(make_tx(200, minutes=3, amount=2.0))
    assert f.small_tx_count_10min == 4
    later = state.features_for(make_tx(201, minutes=30, amount=2.0))
    assert later.small_tx_count_10min == 1


def test_flagged_transactions_do_not_shape_the_baseline():
    state = AccountState("acct-1")
    history(state)
    london = make_tx(50, minutes=5, city="London", lat=51.5, lon=-0.13, amount=60.0)
    state.update(london, learn=False)
    # The genuine cardholder buys at home an hour later: no impossible travel.
    f = state.features_for(make_tx(51, minutes=65))
    assert f.km_from_last < 50
    assert state.n == 10 and state.last_lat == 31.52
    assert state.has_seen(london.transaction_id)


def test_snapshot_roundtrip():
    state = AccountState("acct-1")
    history(state)
    state.update(make_tx(77, amount=1.5))
    restored = AccountState.from_dict(state.to_dict())
    tx = make_tx(78, minutes=1, amount=2.0)
    assert restored.features_for(tx) == state.features_for(tx)
    assert restored.has_seen("tx-acct-1-77")


@pytest.mark.parametrize("features, expected", [
    (Features(small_tx_count_10min=5, tx_count_10min=5), "card_testing"),
    (Features(km_from_last=6000, speed_kmh=12000), "impossible_travel"),
    (Features(amount_zscore=6, amount_ratio_to_mean=9), "amount_spike"),
    (Features(new_category=True, night_hour=True), "night_new_category"),
    (Features(new_category=True), "new_category"),
    (Features(tx_count_10min=7), "high_velocity"),
])
def test_rule_signals(features, expected):
    assert [s.name for s in rule_signals(make_tx(), features)] == [expected]


def test_normal_features_fire_nothing():
    assert rule_signals(make_tx(), Features(tx_count_10min=1, history_size=20)) == []


def test_combine_is_noisy_or():
    signals = [Signal(name="a", detail="", weight=0.5), Signal(name="b", detail="", weight=0.5)]
    assert combine(signals) == 0.75
    assert combine([]) == 0.0


def test_model_requires_fit():
    with pytest.raises(RuntimeError):
        DetectionModel().score(make_tx(), Features())


def test_batch_and_single_scores_match(model):
    items = [(make_tx(i, amount=10 + i * 50), Features(amount_zscore=i, tx_count_10min=1)) for i in range(5)]
    batch = model.score_batch(items)
    assert [model.score(tx, f) for tx, f in items] == batch
    assert all(0 <= ms <= 1 for ms, _, _ in batch)


def test_simulator_is_deterministic_and_ordered():
    a = [e.transaction for e in Simulator(seed=3).stream(500)]
    b = [e.transaction for e in Simulator(seed=3).stream(500)]
    assert a == b
    assert all(x.timestamp <= y.timestamp for x, y in zip(a, a[1:]))


def test_simulator_injects_every_pattern_after_warmup():
    sim = Simulator(seed=5, fraud_rate=0.05)
    events = list(sim.stream(6000))
    assert {e.label for e in events if e.label} == set(PATTERNS)
    assert all(e.transaction.timestamp >= sim.warmup_end for e in events if e.label)
    assert all(e.incident_id for e in events if e.label)


def test_normal_history_has_no_fraud_and_restores_rate():
    sim = Simulator(seed=5, fraud_rate=0.05)
    sim.normal_history(200)
    assert sim.fraud_rate == 0.05
    assert all(e.label is None for e in Simulator(seed=5, fraud_rate=0.0).stream(300))
    assert sim.warmup_end - sim.start == timedelta(hours=48)
