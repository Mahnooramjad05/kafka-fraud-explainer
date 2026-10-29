"""
HTTP API for analysts and monitoring.

  GET /alerts            recent alerts, newest first (filter by min_risk, action)
  GET /alerts/{tx_id}    one alert
  GET /metrics           Prometheus metrics for this process
  GET /health

In Kafka mode a background task consumes the `alerts` topic into an in-memory
ring buffer. Worker processes (scorer, explainer) serve /metrics and /health
with the same app on their own ports.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse

from fraud_explainer import __version__
from fraud_explainer.bus import Bus
from fraud_explainer.config import Settings
from fraud_explainer.metrics import METRICS, Metrics
from fraud_explainer.schemas import Alert

logger = logging.getLogger(__name__)


class AlertStore:
    """Most recent alerts by transaction id (a replayed alert replaces the earlier copy)."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._alerts: OrderedDict[str, Alert] = OrderedDict()

    def add(self, alert: Alert) -> None:
        self._alerts.pop(alert.transaction_id, None)
        self._alerts[alert.transaction_id] = alert
        while len(self._alerts) > self.capacity:
            self._alerts.popitem(last=False)

    def get(self, tx_id: str) -> Optional[Alert]:
        return self._alerts.get(tx_id)

    def recent(self, limit: int, min_risk: float = 0.0, action: Optional[str] = None) -> list[Alert]:
        out = []
        for alert in reversed(self._alerts.values()):
            if alert.risk < min_risk:
                continue
            if action and alert.explanation.recommended_action != action:
                continue
            out.append(alert)
            if len(out) >= limit:
                break
        return out

    def __len__(self) -> int:
        return len(self._alerts)


async def consume_alerts(bus: Bus, settings: Settings, store: AlertStore, stop: asyncio.Event,
                         metrics: Metrics = METRICS) -> None:
    # A unique group per API process, so every instance sees every alert.
    consumer = await bus.consumer([settings.topics.alerts], f"alerts-api-{uuid.uuid4().hex[:8]}")
    try:
        while not stop.is_set():
            for record in await consumer.poll(500, 500):
                try:
                    store.add(Alert.model_validate_json(record.value))
                    metrics.inc("api_alerts_consumed_total", help="Alerts loaded into the API")
                except ValueError as exc:
                    logger.warning("skipping malformed alert: %s", exc)
            await consumer.commit()
    finally:
        await consumer.stop()


def create_app(store: Optional[AlertStore] = None, metrics: Metrics = METRICS,
               bus: Optional[Bus] = None, settings: Optional[Settings] = None,
               role: str = "api") -> FastAPI:
    store = store if store is not None else AlertStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = None
        if bus is not None and settings is not None:
            task = asyncio.create_task(consume_alerts(bus, settings, store, stop, metrics))
        yield
        stop.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=10)

    app = FastAPI(title="Kafka Fraud Explainer", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "role": role, "alerts_loaded": len(store)}

    @app.get("/alerts", response_model=list[Alert])
    def alerts(limit: int = Query(50, ge=1, le=500), min_risk: float = Query(0.0, ge=0, le=1),
               action: Optional[str] = Query(None, pattern="^(approve|review|block)$")) -> list[Alert]:
        return store.recent(limit, min_risk, action)

    @app.get("/alerts/{transaction_id}", response_model=Alert)
    def alert(transaction_id: str) -> Alert:
        found = store.get(transaction_id)
        if found is None:
            raise HTTPException(status_code=404, detail="alert not found")
        return found

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus() -> str:
        return metrics.render()

    return app
