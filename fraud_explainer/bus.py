"""
Messaging layer: a small interface over Kafka, with two implementations.

- KafkaBus     : real Apache Kafka via aiokafka (idempotent producer, manual
                 offset commits, rebalance callbacks, compacted state topic).
- InMemoryBus  : a faithful-enough stand-in (partitions, keys, consumer-group
                 offsets) so the services can be unit-tested without a broker.

Services only use this interface, so the same scorer and explainer code runs
against either.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Protocol

from fraud_explainer.config import Settings

logger = logging.getLogger(__name__)

AssignCallback = Callable[[list[int]], Awaitable[None]]


@dataclass
class Record:
    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: bytes
    headers: dict[str, str] = field(default_factory=dict)


class Producer(Protocol):
    async def send(self, topic: str, value: bytes, key: Optional[str] = None,
                   headers: Optional[dict[str, str]] = None) -> None:
        """Queue a message. Delivery is only guaranteed after flush()."""
    async def flush(self) -> None:
        """Wait until every queued message is acknowledged by the broker."""
    async def stop(self) -> None: ...


class Consumer(Protocol):
    async def poll(self, timeout_ms: int, max_records: int) -> list[Record]: ...
    async def commit(self) -> bool:
        """Commit everything polled so far. Returns False if a rebalance got in the way."""
    async def stop(self) -> None: ...


class Bus(Protocol):
    async def ensure_topics(self) -> None: ...
    async def producer(self) -> Producer: ...
    async def consumer(self, topics: list[str], group: str,
                       on_assign: Optional[AssignCallback] = None) -> Consumer: ...
    async def read_partitions(self, topic: str, partitions: list[int]) -> list[Record]: ...


# --- in-memory ------------------------------------------------------------------


def partition_for(key: Optional[str], partitions: int, counter: int = 0) -> int:
    if key is None:
        return counter % partitions
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % partitions


class InMemoryBus:
    """Topics are lists of partitions; consumer groups track committed offsets per partition."""

    def __init__(self, settings: Optional[Settings] = None, partitions: Optional[int] = None):
        self.settings = settings or Settings()
        self.partitions = partitions or self.settings.partitions
        self.topics: dict[str, list[list[Record]]] = {}
        self.committed: dict[tuple[str, str, int], int] = defaultdict(int)
        self.compacted: set[str] = set()
        self._counter = 0

    def _topic(self, name: str) -> list[list[Record]]:
        if name not in self.topics:
            self.topics[name] = [[] for _ in range(self.partitions)]
        return self.topics[name]

    async def ensure_topics(self) -> None:
        t = self.settings.topics
        for name in (t.raw, t.scored, t.retry, t.dlq, t.alerts, t.account_state):
            self._topic(name)
        self.compacted.add(t.account_state)

    async def producer(self) -> "InMemoryProducer":
        return InMemoryProducer(self)

    async def consumer(self, topics, group, on_assign=None) -> "InMemoryConsumer":
        consumer = InMemoryConsumer(self, topics, group)
        if on_assign is not None:
            await on_assign(list(range(self.partitions)))
        return consumer

    async def read_partitions(self, topic, partitions):
        parts = self._topic(topic)
        return [r for p in partitions for r in parts[p]]

    def append(self, topic: str, value: bytes, key: Optional[str], headers: dict[str, str]) -> Record:
        parts = self._topic(topic)
        self._counter += 1
        p = partition_for(key, self.partitions, self._counter)
        record = Record(topic, p, len(parts[p]), key, value, dict(headers))
        parts[p].append(record)
        return record

    def all(self, topic: str) -> list[Record]:
        return [r for part in self._topic(topic) for r in part]

    def latest_by_key(self, topic: str) -> dict[str, Record]:
        """What a compacted topic converges to: the last record per key."""
        return {r.key: r for r in self.all(topic) if r.key is not None}


class InMemoryProducer:
    def __init__(self, bus: InMemoryBus):
        self.bus = bus

    async def send(self, topic, value, key=None, headers=None):
        self.bus.append(topic, value, key, headers or {})

    async def flush(self):
        pass

    async def stop(self):
        pass


class InMemoryConsumer:
    def __init__(self, bus: InMemoryBus, topics: list[str], group: str):
        self.bus, self.topics, self.group = bus, topics, group
        self.positions = {(t, p): bus.committed[(group, t, p)]
                          for t in topics for p in range(bus.partitions)}

    async def poll(self, timeout_ms, max_records):
        out: list[Record] = []
        progress = True
        while len(out) < max_records and progress:
            progress = False
            for (topic, p), pos in self.positions.items():
                part = self.bus._topic(topic)[p]
                if pos < len(part) and len(out) < max_records:
                    out.append(part[pos])
                    self.positions[(topic, p)] = pos + 1
                    progress = True
        if not out:
            await asyncio.sleep(min(timeout_ms, 10) / 1000)
        return out

    async def commit(self):
        for (topic, p), pos in self.positions.items():
            self.bus.committed[(self.group, topic, p)] = pos
        return True

    async def stop(self):
        pass


# --- Kafka ------------------------------------------------------------------------


class KafkaBus:
    """aiokafka-backed implementation."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def ensure_topics(self) -> None:
        from aiokafka.admin import AIOKafkaAdminClient, NewTopic
        from aiokafka.errors import TopicAlreadyExistsError

        t = self.settings.topics
        n = self.settings.partitions
        new_topics = [NewTopic(name, n, 1) for name in (t.raw, t.scored, t.retry, t.dlq, t.alerts)]
        new_topics.append(NewTopic(t.account_state, n, 1, topic_configs={
            "cleanup.policy": "compact", "min.cleanable.dirty.ratio": "0.1", "segment.ms": "60000",
        }))
        admin = AIOKafkaAdminClient(bootstrap_servers=self.settings.bootstrap_servers)
        await admin.start()
        try:
            for topic in new_topics:
                try:
                    response = await admin.create_topics([topic])
                except TopicAlreadyExistsError:
                    continue
                # create_topics reports per-topic failures in the response instead of raising.
                for name, code, message in [(e[0], e[1], e[2]) for e in response.topic_errors]:
                    if code not in (0, TopicAlreadyExistsError.errno):
                        raise RuntimeError(f"could not create topic {name}: error {code} {message}")
            # Creation is asynchronous on the broker; wait until every topic is visible.
            wanted = {t.name for t in new_topics}
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 15
            while not wanted <= set(await admin.list_topics()):
                if loop.time() > deadline:
                    raise RuntimeError(f"topics not visible after creation: {wanted - set(await admin.list_topics())}")
                await asyncio.sleep(0.2)
        finally:
            await admin.close()

    async def producer(self) -> "KafkaProducer":
        from aiokafka import AIOKafkaProducer

        producer = AIOKafkaProducer(
            bootstrap_servers=self.settings.bootstrap_servers,
            acks="all",
            enable_idempotence=True,
            linger_ms=5,
        )
        await producer.start()
        return KafkaProducer(producer)

    async def consumer(self, topics, group, on_assign=None) -> "KafkaConsumer":
        from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener

        class Listener(ConsumerRebalanceListener):
            async def on_partitions_revoked(self, revoked):
                logger.info("partitions revoked: %s", sorted((tp.topic, tp.partition) for tp in revoked))

            async def on_partitions_assigned(self, assigned):
                logger.info("partitions assigned: %s", sorted((tp.topic, tp.partition) for tp in assigned))
                if on_assign is not None:
                    await on_assign(sorted({tp.partition for tp in assigned if tp.topic == topics[0]}))

        consumer = AIOKafkaConsumer(
            bootstrap_servers=self.settings.bootstrap_servers,
            group_id=group,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await consumer.start()
        consumer.subscribe(topics, listener=Listener())
        return KafkaConsumer(consumer)

    async def read_partitions(self, topic, partitions):
        """Read the given partitions from the beginning to their current end (no group)."""
        from aiokafka import AIOKafkaConsumer, TopicPartition

        if not partitions:
            return []
        consumer = AIOKafkaConsumer(bootstrap_servers=self.settings.bootstrap_servers,
                                    enable_auto_commit=False)
        await consumer.start()
        try:
            tps = [TopicPartition(topic, p) for p in partitions]
            consumer.assign(tps)
            await consumer.seek_to_beginning(*tps)
            end = await consumer.end_offsets(tps)
            out: list[Record] = []
            remaining = [tp for tp in tps if end[tp] > 0]
            while remaining:
                batch = await consumer.getmany(*remaining, timeout_ms=500)
                for msgs in batch.values():
                    out += [_to_record(m) for m in msgs]
                remaining = [tp for tp in remaining if await consumer.position(tp) < end[tp]]
            return out
        finally:
            await consumer.stop()


def _to_record(msg) -> Record:
    return Record(
        topic=msg.topic,
        partition=msg.partition,
        offset=msg.offset,
        key=msg.key.decode("utf-8") if msg.key is not None else None,
        value=msg.value,
        headers={k: v.decode("utf-8") for k, v in (msg.headers or [])},
    )


class KafkaProducer:
    """Batches sends: send() only enqueues; flush() waits for every acknowledgement.

    Waiting per message (send_and_wait) costs a broker round trip each, which made
    a 500-record batch take seconds. Services flush once per batch, before commit.
    """

    def __init__(self, producer):
        self._producer = producer
        self._pending: list = []

    async def send(self, topic, value, key=None, headers=None):
        future = await self._producer.send(
            topic,
            value=value,
            key=key.encode("utf-8") if key is not None else None,
            headers=[(k, v.encode("utf-8")) for k, v in (headers or {}).items()],
        )
        self._pending.append(future)

    async def flush(self):
        pending, self._pending = self._pending, []
        if pending:
            await asyncio.gather(*pending)

    async def stop(self):
        await self.flush()
        await self._producer.stop()


class KafkaConsumer:
    def __init__(self, consumer):
        self._consumer = consumer

    async def poll(self, timeout_ms, max_records):
        batch = await self._consumer.getmany(timeout_ms=timeout_ms, max_records=max_records)
        return [_to_record(m) for msgs in batch.values() for m in msgs]

    async def commit(self):
        # Commits the positions of everything returned by poll() so far. Services call
        # this only after every polled record has been fully handled (at-least-once).
        from aiokafka.errors import CommitFailedError

        try:
            await self._consumer.commit()
            return True
        except CommitFailedError as exc:
            # The group rebalanced mid-batch; the new owner will re-read these records.
            # That is safe: outputs are deduplicated by transaction id downstream.
            logger.warning("offset commit skipped after rebalance: %s", exc)
            return False

    async def stop(self):
        await self._consumer.stop()


def build_bus(settings: Settings) -> Bus:
    return KafkaBus(settings)
