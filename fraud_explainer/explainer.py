"""
Explainer service: transactions.scored (+ retry) -> alerts (+ retry, DLQ).

Only transactions at or above the risk threshold reach the LLM, typically a
few percent of traffic, which is what keeps per-event LLM cost manageable.
Flagged transactions are explained in micro-batches (up to `batch_size`
events, waiting at most `batch_wait_ms` for a batch to fill).

If a batch fails (API error, missing or malformed output), each event is sent
to `transactions.retry` with an attempt counter and a not-before time
(exponential backoff). After `max_attempts` it goes to the dead-letter queue.
Every alert records whether its explanation was faithful: it must cite at
least one signal, and only signals that actually fired.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from fraud_explainer.bus import Bus, Consumer, Producer, Record
from fraud_explainer.config import Settings
from fraud_explainer.llm import Explainer, check_faithfulness
from fraud_explainer.metrics import METRICS, Metrics
from fraud_explainer.schemas import Alert, DeadLetter, ScoredTransaction

logger = logging.getLogger(__name__)

ATTEMPT_HEADER = "x-attempt"
RETRY_AT_HEADER = "x-retry-at"


class RecentIds:
    """Bounded set of recently alerted transaction ids (drops replays within this process)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._ids: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._ids

    def add(self, key: str) -> bool:
        """Return False if the key was already present."""
        if key in self._ids:
            self._ids.move_to_end(key)
            return False
        self._ids[key] = None
        if len(self._ids) > self.capacity:
            self._ids.popitem(last=False)
        return True


class ExplainerService:
    def __init__(self, bus: Bus, settings: Settings, explainer: Explainer,
                 metrics: Metrics = METRICS, clock=time.time, sleep=asyncio.sleep):
        self.bus, self.settings, self.explainer, self.metrics = bus, settings, explainer, metrics
        self.clock, self.sleep = clock, sleep
        self.recent = RecentIds(settings.dedupe_capacity)
        self.producer: Optional[Producer] = None
        self.consumer: Optional[Consumer] = None

    async def start(self) -> None:
        self.producer = await self.bus.producer()
        topics = self.settings.topics
        self.consumer = await self.bus.consumer([topics.scored, topics.retry], self.settings.explainer_group)

    async def _dead_letter(self, record: Record, error: str, attempts: int) -> None:
        letter = DeadLetter(source_topic=record.topic, stage="explainer", error=error, attempts=attempts,
                            key=record.key, payload=record.value.decode("utf-8", errors="replace"),
                            failed_at=datetime.now(timezone.utc))
        await self.producer.send(self.settings.topics.dlq, letter.model_dump_json().encode(), key=record.key)
        self.metrics.inc("dead_letters_total", stage="explainer", help="Messages sent to the dead-letter queue")

    async def _explain_batch(self, batch: list[tuple[Record, ScoredTransaction, int]]) -> None:
        scored = [s for _, s, _ in batch]
        started = time.perf_counter()
        try:
            explanations = await self.explainer.explain(scored)
        except Exception as exc:  # any failure: retry each event with backoff, then DLQ
            self.metrics.inc("llm_batches_total", status="error", help="LLM explanation batches")
            logger.warning("explanation batch of %d failed: %s", len(batch), exc)
            for record, s, attempts in batch:
                await self._retry_or_dead_letter(record, s, attempts, str(exc))
            return
        self.metrics.observe("llm_batch_seconds", time.perf_counter() - started, help="LLM latency per batch")
        self.metrics.inc("llm_batches_total", status="ok", help="LLM explanation batches")

        for (record, s, _), explanation in zip(batch, explanations):
            faithful, unsupported = check_faithfulness(s, explanation)
            tx = s.transaction
            alert = Alert(
                transaction_id=tx.transaction_id, account_id=tx.account_id, amount=tx.amount,
                merchant=tx.merchant, city=tx.city, risk=s.risk, signals=s.signals,
                explanation=explanation, faithful=faithful, unsupported_signals=unsupported,
                explainer=self.explainer.name, created_at=datetime.now(timezone.utc),
            )
            await self.producer.send(self.settings.topics.alerts, alert.model_dump_json().encode(),
                                     key=tx.transaction_id)
            self.recent.add(tx.transaction_id)
            self.metrics.inc("alerts_total", action=explanation.recommended_action, help="Alerts produced")
            if not faithful:
                self.metrics.inc("unfaithful_explanations_total",
                                 help="Explanations citing signals that did not fire")

    async def _retry_or_dead_letter(self, record: Record, scored: ScoredTransaction,
                                    attempts: int, error: str) -> None:
        if attempts >= self.settings.max_attempts:
            await self._dead_letter(record, error, attempts)
            return
        delay = self.settings.retry_backoff_s * 2 ** (attempts - 1)
        await self.producer.send(
            self.settings.topics.retry, scored.model_dump_json().encode(), key=record.key,
            headers={ATTEMPT_HEADER: str(attempts + 1), RETRY_AT_HEADER: f"{self.clock() + delay:.3f}"},
        )
        self.metrics.inc("explain_retries_total", help="Events sent to the retry topic")

    async def run_once(self, timeout_ms: Optional[int] = None) -> int:
        """Poll, explain flagged events in micro-batches, commit. Returns records handled."""
        records = await self.consumer.poll(timeout_ms or self.settings.batch_wait_ms,
                                           self.settings.batch_size * 4)
        if not records:
            return 0

        pending: list[tuple[Record, ScoredTransaction, int]] = []
        for record in records:
            try:
                scored = ScoredTransaction.model_validate_json(record.value)
            except (ValidationError, ValueError) as exc:
                await self._dead_letter(record, f"invalid scored event: {exc}", 1)
                continue
            attempts = int(record.headers.get(ATTEMPT_HEADER, "1"))
            retry_at = float(record.headers.get(RETRY_AT_HEADER, "0"))
            if retry_at > self.clock():
                # Retry topic is low-volume; waiting here is the simplest correct backoff.
                await self.sleep(retry_at - self.clock())
            if scored.risk < self.settings.risk_threshold:
                continue
            if scored.transaction.transaction_id in self.recent:
                self.metrics.inc("explainer_duplicates_total", help="Already-alerted transactions skipped")
                continue
            pending.append((record, scored, attempts))

        for i in range(0, len(pending), self.settings.batch_size):
            await self._explain_batch(pending[i:i + self.settings.batch_size])
        await self.producer.flush()   # outputs must be durable before offsets move
        if not await self.consumer.commit():
            self.metrics.inc("commit_failures_total", help="Offset commits skipped due to a rebalance")
        return len(records)

    async def run(self, stop: asyncio.Event) -> None:
        await self.start()
        try:
            while not stop.is_set():
                await self.run_once()
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self.consumer is not None:
            await self.consumer.stop()
        if self.producer is not None:
            await self.producer.stop()
        await self.explainer.aclose()
