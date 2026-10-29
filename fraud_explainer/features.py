"""
Per-account streaming state and the features computed from it.

Each account's state is small (running amount statistics, recent activity,
last location, seen categories, recent transaction ids) so it can be
snapshotted to the compacted `account-state` topic after every batch and
restored when the scorer is assigned that account's partition.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from fraud_explainer.schemas import Features, Transaction

WINDOW = timedelta(minutes=10)
SMALL_AMOUNT = 5.0
MIN_HISTORY = 5           # transactions needed before amount and category features are trusted
SEEN_IDS_KEPT = 200       # per-account dedupe window
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class AccountState:
    account_id: str
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0                       # Welford running variance accumulator
    last_ts: Optional[datetime] = None
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    recent: deque = field(default_factory=deque)      # (timestamp, amount) within WINDOW
    categories: dict = field(default_factory=dict)    # category -> count
    seen_ids: deque = field(default_factory=lambda: deque(maxlen=SEEN_IDS_KEPT))

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def has_seen(self, transaction_id: str) -> bool:
        return transaction_id in self.seen_ids

    def features_for(self, tx: Transaction) -> Features:
        """Features of `tx` given the history *before* it."""
        window_start = tx.timestamp - WINDOW
        recent = [(ts, amt) for ts, amt in self.recent if ts >= window_start]
        trusted = self.n >= MIN_HISTORY

        zscore, ratio = 0.0, 1.0
        if trusted:
            # Floor the std so a very regular spender doesn't turn small changes into huge z-scores.
            std = max(self.std, 0.25 * self.mean, 1.0)
            zscore = (tx.amount - self.mean) / std
            ratio = tx.amount / self.mean if self.mean > 0 else 1.0

        km, speed = 0.0, 0.0
        if self.last_ts is not None and self.last_lat is not None:
            km = haversine_km(self.last_lat, self.last_lon, tx.lat, tx.lon)
            hours = max((tx.timestamp - self.last_ts).total_seconds() / 3600, 1 / 60)
            speed = km / hours

        return Features(
            amount_zscore=round(zscore, 3),
            amount_ratio_to_mean=round(ratio, 3),
            tx_count_10min=len(recent) + 1,
            small_tx_count_10min=sum(1 for _, a in recent if a < SMALL_AMOUNT) + (tx.amount < SMALL_AMOUNT),
            km_from_last=round(km, 1),
            speed_kmh=round(speed, 1),
            new_category=trusted and tx.merchant_category not in self.categories,
            night_hour=tx.timestamp.hour < 5,
            history_size=self.n,
        )

    def update(self, tx: Transaction, learn: bool = True) -> None:
        """Fold `tx` into the state.

        With learn=False (a transaction the scorer flagged), only the activity window
        and dedupe ids are updated: a suspected-fraud amount, location or category
        must not become part of the account's "normal" baseline. Otherwise the genuine
        cardholder's next purchase at home would look like impossible travel.
        """
        self.recent.append((tx.timestamp, tx.amount))
        while self.recent and self.recent[0][0] < tx.timestamp - WINDOW:
            self.recent.popleft()
        self.seen_ids.append(tx.transaction_id)
        if not learn:
            return
        self.n += 1
        delta = tx.amount - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (tx.amount - self.mean)
        self.last_ts, self.last_lat, self.last_lon = tx.timestamp, tx.lat, tx.lon
        self.categories[tx.merchant_category] = self.categories.get(tx.merchant_category, 0) + 1

    # --- snapshot (de)serialization for the compacted state topic -------------

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "n": self.n,
            "mean": self.mean,
            "m2": self.m2,
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
            "last_lat": self.last_lat,
            "last_lon": self.last_lon,
            "recent": [[ts.isoformat(), amt] for ts, amt in self.recent],
            "categories": self.categories,
            "seen_ids": list(self.seen_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccountState":
        state = cls(
            account_id=data["account_id"],
            n=data["n"],
            mean=data["mean"],
            m2=data["m2"],
            last_ts=datetime.fromisoformat(data["last_ts"]) if data.get("last_ts") else None,
            last_lat=data.get("last_lat"),
            last_lon=data.get("last_lon"),
            categories=dict(data.get("categories", {})),
        )
        state.recent = deque((datetime.fromisoformat(ts), amt) for ts, amt in data.get("recent", []))
        state.seen_ids.extend(data.get("seen_ids", []))
        return state
