"""
Risk scoring: an Isolation Forest for "this looks unlike normal traffic",
plus explicit rule signals for known fraud patterns.

The rules carry the explanation: every signal has a name, a human-readable
detail and a weight. The anomaly model contributes its own signal when its
score is high, so a transaction can't be flagged without at least one named
reason the LLM is allowed to cite.

    risk = 1 - prod(1 - weight)   over all fired signals
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.ensemble import IsolationForest

from fraud_explainer.features import AccountState
from fraud_explainer.schemas import Features, Signal, Transaction

MODEL_SIGNAL_THRESHOLD = 0.8


def feature_vector(tx: Transaction, f: Features) -> list[float]:
    return [
        math.log1p(tx.amount),
        max(min(f.amount_zscore, 20.0), -5.0),
        float(f.tx_count_10min),
        float(f.small_tx_count_10min),
        math.log1p(f.speed_kmh),
        float(f.new_category),
        float(f.night_hour),
    ]


def rule_signals(tx: Transaction, f: Features) -> list[Signal]:
    signals = []
    if f.small_tx_count_10min >= 4:
        signals.append(Signal(name="card_testing", weight=0.85, detail=(
            f"{f.small_tx_count_10min} charges under $5 in the last 10 minutes")))
    if f.km_from_last > 500 and f.speed_kmh > 900:
        signals.append(Signal(name="impossible_travel", weight=0.9, detail=(
            f"{f.km_from_last:,.0f} km from the previous transaction, "
            f"implying {f.speed_kmh:,.0f} km/h")))
    if f.amount_zscore >= 4 and f.amount_ratio_to_mean >= 5:
        signals.append(Signal(name="amount_spike", weight=0.75, detail=(
            f"${tx.amount:,.2f} is {f.amount_ratio_to_mean:.1f}x this account's average")))
    if f.new_category and f.night_hour:
        signals.append(Signal(name="night_new_category", weight=0.7, detail=(
            f"first-ever '{tx.merchant_category}' purchase, at "
            f"{tx.timestamp.strftime('%H:%M')} UTC")))
    elif f.new_category:
        signals.append(Signal(name="new_category", weight=0.25, detail=(
            f"first-ever '{tx.merchant_category}' purchase on this account")))
    if f.tx_count_10min >= 6 and f.small_tx_count_10min < 4:
        signals.append(Signal(name="high_velocity", weight=0.5, detail=(
            f"{f.tx_count_10min} transactions in the last 10 minutes")))
    return signals


def combine(signals: Iterable[Signal]) -> float:
    remaining = 1.0
    for s in signals:
        remaining *= 1.0 - s.weight
    return round(1.0 - remaining, 4)


class DetectionModel:
    def __init__(self, n_estimators: int = 150, seed: int = 7):
        self.forest = IsolationForest(n_estimators=n_estimators, random_state=seed)
        self._low = 0.0
        self._high = 1.0
        self.fitted = False

    def fit(self, normal_history: list[Transaction]) -> "DetectionModel":
        """Train on fraud-free traffic, replaying it through account state to get features."""
        states: dict[str, AccountState] = {}
        rows = []
        for tx in sorted(normal_history, key=lambda t: t.timestamp):
            state = states.setdefault(tx.account_id, AccountState(tx.account_id))
            rows.append(feature_vector(tx, state.features_for(tx)))
            state.update(tx)
        x = np.array(rows)
        self.forest.fit(x)
        raw = -self.forest.score_samples(x)
        # Scale so typical normal traffic maps near 0 and the rarest 0.1% near 1.
        self._low, self._high = float(np.percentile(raw, 50)), float(np.percentile(raw, 99.9))
        self.fitted = True
        return self

    def anomaly_scores(self, items: list[tuple[Transaction, Features]]) -> list[float]:
        """Scaled anomaly scores for a batch. One vectorized call: scoring rows one at a
        time costs ~20 ms each in scikit-learn, far too slow for a stream."""
        if not items:
            return []
        raw = -self.forest.score_samples(np.array([feature_vector(tx, f) for tx, f in items]))
        scaled = (raw - self._low) / max(self._high - self._low, 1e-9)
        return [round(float(v), 4) for v in np.clip(scaled, 0.0, 1.0)]

    def score_batch(self, items: list[tuple[Transaction, Features]]) -> list[tuple[float, float, list[Signal]]]:
        """Return (model_score, risk, signals) for each (transaction, features) pair."""
        if not self.fitted:
            raise RuntimeError("DetectionModel.fit() must be called before scoring")
        results = []
        for (tx, f), model_score in zip(items, self.anomaly_scores(items)):
            signals = rule_signals(tx, f)
            if model_score >= MODEL_SIGNAL_THRESHOLD:
                signals.append(Signal(name="anomaly_model", weight=0.4, detail=(
                    f"anomaly score {model_score:.2f}: this combination of amount, timing and "
                    "activity is rare in normal traffic")))
            results.append((model_score, combine(signals), signals))
        return results

    def score(self, tx: Transaction, f: Features) -> tuple[float, float, list[Signal]]:
        return self.score_batch([(tx, f)])[0]
