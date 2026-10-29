import os
from datetime import datetime, timedelta, timezone

import pytest

from fraud_explainer.bus import InMemoryBus
from fraud_explainer.config import Settings
from fraud_explainer.metrics import Metrics
from fraud_explainer.model import DetectionModel
from fraud_explainer.schemas import Transaction
from fraud_explainer.simulator import Simulator

# Never call a real LLM from tests.
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

T0 = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)


def make_tx(i: int = 1, account: str = "acct-1", minutes: float = 0, amount: float = 40.0,
            category: str = "groceries", city: str = "Lahore", lat: float = 31.52, lon: float = 74.36,
            ts: datetime | None = None) -> Transaction:
    return Transaction(
        transaction_id=f"tx-{account}-{i}", account_id=account,
        timestamp=ts or T0 + timedelta(minutes=minutes), amount=amount,
        merchant="FreshMart", merchant_category=category, city=city, lat=lat, lon=lon,
    )


@pytest.fixture(scope="session")
def model() -> DetectionModel:
    return DetectionModel(n_estimators=80).fit(Simulator(seed=1, fraud_rate=0).normal_history(3000))


@pytest.fixture
def settings() -> Settings:
    return Settings(batch_size=4, retry_backoff_s=0.0)


@pytest.fixture
async def bus(settings) -> InMemoryBus:
    b = InMemoryBus(settings)
    await b.ensure_topics()
    return b


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()
