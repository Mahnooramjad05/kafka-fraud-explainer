"""Publish simulated transactions to `transactions.raw`, keyed by account_id."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fraud_explainer.bus import Bus
from fraud_explainer.config import Settings
from fraud_explainer.simulator import LabeledTransaction, Simulator

logger = logging.getLogger(__name__)


async def publish(bus: Bus, settings: Settings, simulator: Simulator, num_events: int,
                  rate_per_s: float = 0.0, malformed_every: Optional[int] = None) -> list[LabeledTransaction]:
    """Send `num_events` events; rate 0 means as fast as possible.

    `malformed_every` injects an invalid message every N events, to exercise the
    dead-letter path. Returns what was sent (with labels) for evaluation.
    """
    producer = await bus.producer()
    sent: list[LabeledTransaction] = []
    try:
        for i, event in enumerate(simulator.stream(num_events), start=1):
            tx = event.transaction
            await producer.send(settings.topics.raw, tx.model_dump_json().encode(), key=tx.account_id)
            sent.append(event)
            if malformed_every and i % malformed_every == 0:
                await producer.send(settings.topics.raw, b'{"transaction_id": "broken", "amount": -1}',
                                    key=tx.account_id)
            if rate_per_s > 0:
                await producer.flush()
                await asyncio.sleep(1 / rate_per_s)
            elif i % 500 == 0:
                await producer.flush()
            if i % 1000 == 0:
                logger.info("published %d events", i)
    finally:
        await producer.stop()
    return sent
