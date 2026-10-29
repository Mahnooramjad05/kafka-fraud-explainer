"""
End-to-end tests against a real Kafka broker.

Skipped unless a broker is reachable at KAFKA_BOOTSTRAP_SERVERS (default
127.0.0.1:9092); CI sets REQUIRE_KAFKA=1 so a missing broker fails instead.
Start one locally with:

    docker run -d -p 9092:9092 apache/kafka:3.8.0

Each test uses its own uniquely prefixed topics and consumer groups.
"""

import asyncio
import dataclasses
import os
import socket
import uuid

import pytest

from fraud_explainer.bus import KafkaBus
from fraud_explainer.config import Settings, Topics
from fraud_explainer.explainer import ExplainerService
from fraud_explainer.llm import MockExplainer
from fraud_explainer.metrics import Metrics
from fraud_explainer.producer import publish
from fraud_explainer.schemas import Alert, DeadLetter, ScoredTransaction
from fraud_explainer.scorer import ScorerService
from fraud_explainer.simulator import Simulator

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")


def _reachable() -> bool:
    host, _, port = BOOTSTRAP.split(",")[0].rpartition(":")
    try:
        socket.create_connection((host, int(port)), timeout=2).close()
        return True
    except OSError:
        return False


if not _reachable():
    if os.getenv("REQUIRE_KAFKA") == "1":
        raise RuntimeError(f"REQUIRE_KAFKA=1 but no broker at {BOOTSTRAP}")
    pytestmark = pytest.mark.skip(reason=f"no Kafka broker at {BOOTSTRAP}")
else:
    pytestmark = pytest.mark.kafka


@pytest.fixture
def kafka_settings() -> Settings:
    run = f"it-{uuid.uuid4().hex[:8]}"
    return dataclasses.replace(
        Settings(bootstrap_servers=BOOTSTRAP, batch_size=8, retry_backoff_s=0.0),
        topics=Topics.prefixed(run), scorer_group=f"{run}-scorer", explainer_group=f"{run}-explainer",
    )


async def read_all(bus: KafkaBus, topic: str, partitions: int) -> list:
    return await bus.read_partitions(topic, list(range(partitions)))


async def run_until(service, done, timeout_s: float = 90.0) -> None:
    """Poll the service until `done()` is true (or fail after timeout)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not await done():
        assert loop.time() < deadline, "timed out waiting for the pipeline"
        await service.run_once(timeout_ms=500)


async def test_full_pipeline_on_kafka(kafka_settings, model):
    s = kafka_settings
    bus = KafkaBus(s)
    await bus.ensure_topics()
    metrics = Metrics()

    # Few accounts and a short warm-up, so fraud starts within the first 1,500 events.
    sim = Simulator(seed=21, num_accounts=50, warmup_hours=6, fraud_rate=0.05)
    sent = await publish(bus, s, sim, 1500, malformed_every=500)
    assert any(e.label for e in sent)
    valid = len(sent)

    scorer = ScorerService(bus, s, model, metrics)
    await scorer.start()

    async def all_scored():
        return metrics.get("transactions_scored_total") >= valid
    await run_until(scorer, all_scored)
    await scorer.stop()

    scored = [ScoredTransaction.model_validate_json(r.value) for r in await read_all(bus, s.topics.scored, s.partitions)]
    flagged = [x for x in scored if x.risk >= s.risk_threshold]
    assert len(scored) == valid
    assert flagged, "simulated fraud should be flagged"

    letters = [DeadLetter.model_validate_json(r.value) for r in await read_all(bus, s.topics.dlq, s.partitions)]
    assert len(letters) == 3 and all(d.stage == "scorer" for d in letters)

    explainer = ExplainerService(bus, s, MockExplainer(), metrics)
    await explainer.start()

    async def all_alerted():
        return sum(v for (n, _), v in metrics.counters.items() if n == "alerts_total") >= len(flagged)
    await run_until(explainer, all_alerted)
    await explainer.stop()

    alerts = [Alert.model_validate_json(r.value) for r in await read_all(bus, s.topics.alerts, s.partitions)]
    assert {a.transaction_id for a in alerts} == {x.transaction.transaction_id for x in flagged}
    assert all(a.faithful for a in alerts)


async def test_state_topic_is_compacted_and_restored_after_restart(kafka_settings, model):
    from aiokafka.admin import AIOKafkaAdminClient
    from aiokafka.admin.config_resource import ConfigResource, ConfigResourceType

    s = kafka_settings
    bus = KafkaBus(s)
    await bus.ensure_topics()

    admin = AIOKafkaAdminClient(bootstrap_servers=s.bootstrap_servers)
    await admin.start()
    try:
        [response] = await admin.describe_configs([ConfigResource(ConfigResourceType.TOPIC, s.topics.account_state)])
        configs = {entry[0]: entry[1] for entry in response.resources[0][4]}
        assert configs["cleanup.policy"] == "compact"
    finally:
        await admin.close()

    sent = await publish(bus, s, Simulator(seed=22, num_accounts=20, fraud_rate=0.0), 300)
    metrics = Metrics()
    first = ScorerService(bus, s, model, metrics)
    await first.start()

    async def done():
        return metrics.get("transactions_scored_total") >= len(sent)
    await run_until(first, done)
    history = {a: st.n for a, st in first.states.items()}
    await first.stop()

    # A new instance in the same group: offsets are committed, state comes back from Kafka.
    restarted_metrics = Metrics()
    second = ScorerService(bus, s, model, restarted_metrics)
    await second.start()
    replay = sent[0].transaction
    producer = await bus.producer()
    await producer.send(s.topics.raw, replay.model_dump_json().encode(), key=replay.account_id)
    await producer.stop()

    async def replay_seen():
        return restarted_metrics.get("scorer_duplicates_total") >= 1
    await run_until(second, replay_seen)
    await second.stop()

    assert restarted_metrics.get("transactions_scored_total") == 0   # nothing reprocessed
    assert restarted_metrics.get("scorer_state_restored_total") >= len(history)
    assert {a: st.n for a, st in second.states.items()} == history
