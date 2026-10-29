"""
Scorer service: transactions.raw -> transactions.scored (+ account-state, DLQ).

For each polled batch:
  1. validate every message (malformed ones go to the dead-letter queue);
  2. drop duplicates (transaction ids already seen for that account);
  3. compute features from per-account state, in order, and update the state;
  4. score the whole batch with one vectorized model call;
  5. produce scored events, then a state snapshot per touched account to the
     compacted `account-state` topic;
  6. commit offsets.

Committing last gives at-least-once delivery; the per-account dedupe ids live
in the snapshot, so a replay after a crash does not emit duplicates. When
partitions are assigned (startup or rebalance), state for those partitions is
restored from `account-state`, which is co-partitioned with the input topic
(same partition count, same account_id keys).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from fraud_explainer.bus import Bus, Consumer, Producer, Record
from fraud_explainer.config import Settings
from fraud_explainer.features import AccountState
from fraud_explainer.metrics import METRICS, Metrics
from fraud_explainer.model import DetectionModel, combine, rule_signals
from fraud_explainer.schemas import DeadLetter, ScoredTransaction, Transaction

logger = logging.getLogger(__name__)


def score_transactions(states: dict[str, AccountState], model: DetectionModel,
                       txs: list[Transaction], threshold: float) -> list[Optional[ScoredTransaction]]:
    """Score transactions in order; returns None for duplicates.

    State updates use the rule-based risk to decide whether a transaction may
    shape the account's baseline (suspected fraud must not). The anomaly model
    then scores the whole batch in one call.
    """
    out: list[Optional[ScoredTransaction]] = [None] * len(txs)
    items, positions = [], []
    for i, tx in enumerate(txs):
        state = states.setdefault(tx.account_id, AccountState(tx.account_id))
        if state.has_seen(tx.transaction_id):
            continue
        features = state.features_for(tx)
        state.update(tx, learn=combine(rule_signals(tx, features)) < threshold)
        items.append((tx, features))
        positions.append(i)

    now = datetime.now(timezone.utc)
    for i, (tx, features), (model_score, risk, signals) in zip(positions, items, model.score_batch(items)):
        out[i] = ScoredTransaction(transaction=tx, features=features, model_score=model_score,
                                   risk=risk, signals=signals, scored_at=now)
    return out


class ScorerService:
    def __init__(self, bus: Bus, settings: Settings, model: DetectionModel,
                 metrics: Metrics = METRICS, max_records: int = 500):
        self.bus, self.settings, self.model, self.metrics = bus, settings, model, metrics
        self.max_records = max_records
        self.states: dict[str, AccountState] = {}
        # aiokafka runs the rebalance callback concurrently with fetching, so records from
        # newly assigned partitions can arrive while their state is still being restored.
        # restore() takes this lock first thing; batches wait for it before touching state.
        self._state_lock = asyncio.Lock()
        self.producer: Optional[Producer] = None
        self.consumer: Optional[Consumer] = None

    async def start(self) -> None:
        self.producer = await self.bus.producer()
        self.consumer = await self.bus.consumer(
            [self.settings.topics.raw], self.settings.scorer_group, on_assign=self.restore
        )

    async def restore(self, partitions: list[int]) -> None:
        """Rebuild account state for newly assigned partitions from the compacted topic."""
        async with self._state_lock:  # acquired before the first await: no batch can interleave
            records = await self.bus.read_partitions(self.settings.topics.account_state, partitions)
            restored = 0
            for record in records:  # later snapshots overwrite earlier ones
                if record.key is None:
                    continue
                if not record.value:
                    self.states.pop(record.key, None)  # tombstone
                    continue
                self.states[record.key] = AccountState.from_dict(json.loads(record.value))
                restored += 1
        logger.info("restored %d state snapshots for partitions %s", restored, partitions)
        self.metrics.inc("scorer_state_restored_total", restored, help="Account-state snapshots restored")

    async def _dead_letter(self, record: Record, error: str) -> None:
        letter = DeadLetter(source_topic=record.topic, stage="scorer", error=error, key=record.key,
                            payload=record.value.decode("utf-8", errors="replace"),
                            failed_at=datetime.now(timezone.utc))
        await self.producer.send(self.settings.topics.dlq, letter.model_dump_json().encode(), key=record.key)
        self.metrics.inc("dead_letters_total", stage="scorer", help="Messages sent to the dead-letter queue")

    async def run_once(self, timeout_ms: int = 500) -> int:
        """Process one polled batch. Returns the number of records handled."""
        records = await self.consumer.poll(timeout_ms, self.max_records)
        if not records:
            return 0
        async with self._state_lock:
            return await self._process(records)

    async def _process(self, records: list[Record]) -> int:
        started = time.perf_counter()

        valid: list[tuple[Record, Transaction]] = []
        for record in records:
            try:
                valid.append((record, Transaction.model_validate_json(record.value)))
            except (ValidationError, ValueError) as exc:
                await self._dead_letter(record, f"invalid transaction: {exc}")

        scored = score_transactions(self.states, self.model, [tx for _, tx in valid],
                                    self.settings.risk_threshold)
        touched = set()
        for (record, tx), result in zip(valid, scored):
            if result is None:
                self.metrics.inc("scorer_duplicates_total", help="Duplicate transactions dropped")
                continue
            await self.producer.send(self.settings.topics.scored, result.model_dump_json().encode(),
                                     key=tx.account_id)
            touched.add(tx.account_id)
            self.metrics.inc("transactions_scored_total", help="Transactions scored")
            if result.risk >= self.settings.risk_threshold:
                self.metrics.inc("transactions_flagged_total", help="Transactions at or above the risk threshold")

        for account_id in touched:
            await self.producer.send(self.settings.topics.account_state,
                                     json.dumps(self.states[account_id].to_dict()).encode(),
                                     key=account_id)
        await self.producer.flush()   # outputs must be durable before offsets move
        if not await self.consumer.commit():
            self.metrics.inc("commit_failures_total", help="Offset commits skipped due to a rebalance")
        self.metrics.observe("scorer_batch_seconds", time.perf_counter() - started,
                             help="Time to score one polled batch")
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
