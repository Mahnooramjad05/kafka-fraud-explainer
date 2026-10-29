"""Settings, read from environment variables with sensible local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


@dataclass(frozen=True)
class Topics:
    raw: str = "transactions.raw"
    scored: str = "transactions.scored"
    retry: str = "transactions.retry"
    dlq: str = "transactions.dlq"
    alerts: str = "alerts"
    account_state: str = "account-state"

    @classmethod
    def prefixed(cls, prefix: str) -> "Topics":
        """Same topic set under a prefix, e.g. for isolated integration-test runs."""
        base = cls()
        return cls(**{name: f"{prefix}.{value}" for name, value in vars(base).items()})


@dataclass(frozen=True)
class Settings:
    # 127.0.0.1 rather than localhost: on Windows, "localhost" tries IPv6 first and each
    # connection to a Docker-published port stalls ~20 s before falling back.
    bootstrap_servers: str = "127.0.0.1:9092"
    partitions: int = 3
    topics: Topics = field(default_factory=Topics)
    scorer_group: str = "fraud-scorer"
    explainer_group: str = "fraud-explainer"
    risk_threshold: float = 0.6
    batch_size: int = 8
    batch_wait_ms: int = 200
    max_attempts: int = 3
    retry_backoff_s: float = 1.0
    llm_provider: str = "mock"
    dedupe_capacity: int = 100_000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            bootstrap_servers=_env("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092"),
            partitions=int(_env("KAFKA_PARTITIONS", "3")),
            risk_threshold=float(_env("RISK_THRESHOLD", "0.6")),
            batch_size=int(_env("EXPLAIN_BATCH_SIZE", "8")),
            batch_wait_ms=int(_env("EXPLAIN_BATCH_WAIT_MS", "200")),
            max_attempts=int(_env("EXPLAIN_MAX_ATTEMPTS", "3")),
            retry_backoff_s=float(_env("EXPLAIN_RETRY_BACKOFF_S", "1.0")),
            llm_provider=_env("LLM_PROVIDER", "mock").lower(),
        )
