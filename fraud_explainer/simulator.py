"""
Synthetic card-transaction stream with labeled fraud patterns.

Normal behavior: each account has a home city, a few usual merchant
categories, a typical spend (log-normal), and makes a purchase every few
hours of simulated time. After a warm-up period, fraud incidents are
injected at a configurable rate:

  - card_testing        : a burst of 5-8 charges under $5 within a few minutes
  - impossible_travel   : a purchase thousands of km away shortly after a normal one
  - amount_spike        : a purchase 12-25x the account's typical amount
  - night_new_category  : a purchase in a category the account never uses, at 1-4am

Each event is yielded with its label (None for normal), so detection quality
can be measured exactly. Labels are never sent to the scorer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

from fraud_explainer.schemas import Transaction

CITIES = {
    "Islamabad": (33.684, 73.048), "Karachi": (24.861, 67.010), "Lahore": (31.520, 74.359),
    "Dubai": (25.205, 55.271), "London": (51.507, -0.128), "New York": (40.713, -74.006),
    "Toronto": (43.653, -79.383), "Singapore": (1.352, 103.820), "Sydney": (-33.869, 151.209),
    "Berlin": (52.520, 13.405), "Sao Paulo": (-23.551, -46.633), "Nairobi": (-1.292, 36.822),
}
CATEGORIES = {
    "groceries": ["FreshMart", "GreenGrocer", "DailyBasket"],
    "restaurants": ["Spice Route", "Urban Diner", "Noodle Bar"],
    "fuel": ["FuelGo", "PetroPoint"],
    "pharmacy": ["CarePlus Pharmacy", "MediCorner"],
    "clothing": ["ThreadLine", "StyleHub"],
    "transport": ["CityRide", "MetroCard Topup"],
    "online_retail": ["ShopSphere", "QuickCart"],
    "utilities": ["PowerGrid Bill", "NetFiber"],
}
RISKY_CATEGORIES = {
    "crypto_exchange": ["CoinVault", "BitHarbor"],
    "wire_transfer": ["FastWire", "GlobalRemit"],
    "gift_cards": ["GiftCardDepot"],
}
SPIKE_CATEGORIES = {"electronics": ["GadgetWorld", "MegaTech"], "jewelry": ["GoldCrest Jewelers"]}
PATTERNS = ("card_testing", "impossible_travel", "amount_spike", "night_new_category")


@dataclass
class Account:
    account_id: str
    home: str
    categories: list[str]
    mu: float          # log-normal location of typical spend
    sigma: float
    next_at: datetime


@dataclass
class LabeledTransaction:
    transaction: Transaction
    label: Optional[str]          # fraud pattern, or None for normal
    incident_id: Optional[str]    # groups the events of one fraud incident


class Simulator:
    def __init__(self, num_accounts: int = 200, seed: int = 7,
                 start: Optional[datetime] = None, fraud_rate: float = 0.02,
                 warmup_hours: float = 48.0):
        self.rng = random.Random(seed)
        self.start = start or datetime(2026, 1, 5, tzinfo=timezone.utc)
        self.fraud_rate = fraud_rate
        self.warmup_end = self.start + timedelta(hours=warmup_hours)
        self._tx_counter = 0
        self._incident_counter = 0
        self.accounts = [self._make_account(i) for i in range(num_accounts)]

    def _make_account(self, i: int) -> Account:
        rng = self.rng
        return Account(
            account_id=f"acct-{i:04d}",
            home=rng.choice(list(CITIES)),
            categories=rng.sample(list(CATEGORIES), k=rng.randint(3, 5)),
            mu=math.log(rng.uniform(15, 120)),
            sigma=rng.uniform(0.3, 0.6),
            next_at=self.start + timedelta(minutes=rng.uniform(0, 240)),
        )

    def _tx(self, account: Account, ts: datetime, amount: float, category: str,
            merchant: str, city: str) -> Transaction:
        self._tx_counter += 1
        lat, lon = CITIES[city]
        return Transaction(
            transaction_id=f"tx-{self._tx_counter:08d}",
            account_id=account.account_id,
            timestamp=ts,
            amount=round(max(amount, 0.5), 2),
            merchant=merchant,
            merchant_category=category,
            city=city,
            lat=round(lat + self.rng.uniform(-0.05, 0.05), 4),
            lon=round(lon + self.rng.uniform(-0.05, 0.05), 4),
        )

    def _normal(self, account: Account, ts: datetime) -> Transaction:
        category = self.rng.choice(account.categories)
        amount = self.rng.lognormvariate(account.mu, account.sigma)
        merchant = self.rng.choice(CATEGORIES[category])
        return self._tx(account, ts, amount, category, merchant, account.home)

    def _incident(self, account: Account, ts: datetime) -> list[LabeledTransaction]:
        rng = self.rng
        pattern = rng.choice(PATTERNS)
        self._incident_counter += 1
        incident = f"inc-{self._incident_counter:05d}"
        typical = math.exp(account.mu)
        events: list[Transaction] = []

        if pattern == "card_testing":
            t = ts
            for _ in range(rng.randint(5, 8)):
                t += timedelta(seconds=rng.uniform(15, 60))
                events.append(self._tx(account, t, rng.uniform(0.5, 4.99), "online_retail",
                                       rng.choice(["QuickCart", "ShopSphere"]), account.home))
        elif pattern == "impossible_travel":
            far = [c for c in CITIES if _distance(c, account.home) > 3000]
            city = rng.choice(far)
            category = rng.choice(account.categories)
            events.append(self._tx(account, ts + timedelta(minutes=rng.uniform(5, 40)),
                                   typical * rng.uniform(1.0, 3.0), category,
                                   rng.choice(CATEGORIES[category]), city))
        elif pattern == "amount_spike":
            category = rng.choice(list(SPIKE_CATEGORIES))
            events.append(self._tx(account, ts, typical * rng.uniform(12, 25), category,
                                   rng.choice(SPIKE_CATEGORIES[category]), account.home))
        else:  # night_new_category
            category = rng.choice(list(RISKY_CATEGORIES))
            night = (ts + timedelta(days=1)).replace(hour=rng.randint(1, 3), minute=rng.randint(0, 59))
            events.append(self._tx(account, night, typical * rng.uniform(2, 5), category,
                                   rng.choice(RISKY_CATEGORIES[category]), account.home))
        return [LabeledTransaction(tx, pattern, incident) for tx in events]

    def stream(self, num_events: int) -> Iterator[LabeledTransaction]:
        """Yield events in timestamp order (fraud events slot in at their own times)."""
        pending: list[LabeledTransaction] = []
        produced = 0
        while produced < num_events:
            account = min(self.accounts, key=lambda a: a.next_at)
            ts = account.next_at
            # Flush any queued fraud events that happen before this normal one.
            pending.sort(key=lambda e: e.transaction.timestamp)
            while pending and pending[0].transaction.timestamp <= ts and produced < num_events:
                yield pending.pop(0)
                produced += 1
            if produced >= num_events:
                break

            yield LabeledTransaction(self._normal(account, ts), None, None)
            produced += 1
            # Daytime-heavy spending: next purchase in 1-6 hours, nudged out of the night.
            nxt = ts + timedelta(hours=self.rng.uniform(1, 6))
            if nxt.hour < 6:
                nxt = nxt.replace(hour=self.rng.randint(7, 10))
            account.next_at = nxt

            if ts >= self.warmup_end and self.rng.random() < self.fraud_rate:
                pending.extend(self._incident(account, ts))

    def normal_history(self, num_events: int) -> list[Transaction]:
        """Fraud-free traffic, used to train the anomaly model."""
        saved = self.fraud_rate
        self.fraud_rate = 0.0
        try:
            return [e.transaction for e in self.stream(num_events)]
        finally:
            self.fraud_rate = saved


def _distance(a: str, b: str) -> float:
    from fraud_explainer.features import haversine_km

    return haversine_km(*CITIES[a], *CITIES[b])
