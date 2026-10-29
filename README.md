# kafka-fraud-explainer

[![ci](https://github.com/Mahnooramjad05/kafka-fraud-explainer/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahnooramjad05/kafka-fraud-explainer/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![kafka](https://img.shields.io/badge/Apache%20Kafka-3.8-black)
![license](https://img.shields.io/badge/license-MIT-green)

Real-time fraud detection on Apache Kafka. Every card transaction is scored as it streams in, using
per-account features, an Isolation Forest and explicit fraud rules. The few that cross the risk threshold
are explained by an LLM in language an analyst can act on, and each explanation is checked against the
signals that actually fired.

```
 simulator ──▶ transactions.raw ──▶ [scorer] ──▶ transactions.scored ──▶ [explainer] ──▶ alerts ──▶ [API]
 (keyed by        (ordered per       features,         risk ≥ threshold      LLM, batched,             /alerts
  account_id)      account)          Isolation Forest   only (~4% here)       faithfulness check        /metrics
                                     + rule signals                              │
                                          │                                       ▼
                                          ▼                                transactions.retry ──▶ transactions.dlq
                                   account-state                           (backoff, max attempts)
                                   (compacted, co-partitioned,
                                    restored on rebalance)
```

## Why it's built this way

- **The LLM only sees what matters.** Scoring is cheap and runs on everything; the LLM runs only on flagged
  transactions (3.6% of traffic in the evaluation below), in micro-batches. That's what makes per-event LLM
  explanations affordable on a stream.
- **Explanations must cite real signals.** Every flag carries named signals (`card_testing`,
  `impossible_travel`, `amount_spike`, ...). The LLM is told to cite only those, and every alert records
  whether it did (`faithful`, `unsupported_signals`).
- **State survives restarts and rebalances.** Per-account features (running mean/variance, 10-minute
  activity window, last location, known categories, recent transaction ids) are snapshotted to a compacted
  topic that is co-partitioned with the input. When a scorer is assigned partitions, it restores exactly
  those accounts' state.
- **At-least-once, without duplicate alerts.** Outputs are flushed before offsets are committed. Transaction
  ids live in the account snapshot, so a replay after a crash is recognized and dropped.
- **Suspected fraud doesn't poison the baseline.** A flagged transaction updates the activity window, but
  not the account's normal amount, location or categories. Otherwise the real cardholder's next purchase at
  home would look like impossible travel.
- **Failures are contained.** Malformed events go to a dead-letter queue. Failed LLM batches are retried
  through a retry topic with exponential backoff, then dead-lettered.

## Quickstart

```bash
docker compose up --build
```

This starts Kafka (KRaft, single node), the scorer, the explainer, the API and a simulator that publishes
5,000 transactions with injected fraud. Then:

```bash
curl "http://localhost:8000/alerts?limit=3"
curl "http://localhost:8000/alerts?action=block&min_risk=0.9"
curl http://localhost:8002/metrics          # explainer: LLM batches, latency, alerts, retries
```

If 8000 or 8002 is taken on your machine, set `API_PORT` / `EXPLAINER_METRICS_PORT` before `docker compose up`.

Scale the scorer to watch partitions rebalance and state restore from the compacted topic:

```bash
docker compose up --build --scale scorer=2
```

Use a real LLM instead of the offline mock:

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... docker compose up --build
# or LLM_PROVIDER=openai OPENAI_API_KEY=...
```

## Evaluation

The simulator labels every injected fraud event, so detection can be measured exactly:

```bash
python -m eval.detection --events 10000
```

```
Events: 10000   fraud events: 506   flagged: 356
Transaction level  precision 0.992  recall 0.698  F1 0.819
Incident recall    1.000  (incident caught if any of its events is flagged)

Per pattern              event recall   incident recall
  card_testing                0.525            1.000
  impossible_travel           1.000            1.000
  amount_spike                1.000            1.000
  night_new_category          1.000            1.000

Sent to LLM: 3.6% of traffic   alerts: 356   faithful explanations: 100.0%
```

How to read this honestly:

- The traffic is **synthetic**, and the rules were written for the same four patterns the simulator
  injects, so these numbers show the pipeline works end to end. They are not a claim about real-world
  fraud detection.
- Card-testing *event* recall is about 50% by design: a burst only becomes detectable after four small
  charges. Every burst is still caught (incident recall 1.0).
- 100% faithfulness is with the deterministic mock explainer, which cites exactly the fired signals. With a
  real model, this is the number to watch.

## Fraud patterns and signals

| Pattern (simulator) | Signal (scorer) | Fires when |
|---|---|---|
| Card testing: 5–8 charges under $5 within minutes | `card_testing` | ≥ 4 charges under $5 in 10 minutes |
| Impossible travel: purchase thousands of km away | `impossible_travel` | > 500 km from the last purchase at > 900 km/h |
| Amount spike: 12–25× the usual amount | `amount_spike` | z-score ≥ 4 and ≥ 5× the account's mean |
| New risky category at night | `night_new_category` | first purchase in a category, between 00:00 and 05:00 |
| (any) | `anomaly_model` | Isolation Forest score ≥ 0.8 |
| (any) | `high_velocity`, `new_category` | weaker supporting signals |

Risk combines signal weights as `1 - Π(1 - weight)`, so a transaction can't be flagged without at least one
named reason.

## Project structure

```
fraud_explainer/
  bus.py          Kafka (aiokafka) and in-memory implementations of one small interface
  scorer.py       transactions.raw -> transactions.scored, state snapshots, DLQ
  explainer.py    transactions.scored -> alerts, micro-batching, retry topic, DLQ
  features.py     per-account streaming state and features
  model.py        Isolation Forest + rule signals
  llm.py          mock / OpenAI / Anthropic explainers, faithfulness check
  simulator.py    labeled synthetic transaction stream
  api.py          FastAPI: /alerts, /metrics, /health
  __main__.py     simulate | scorer | explainer | api
eval/detection.py end-to-end evaluation on labeled traffic
tests/            unit tests (in-memory bus) + integration tests against real Kafka
```

## Tests

```bash
pip install -e ".[llm,dev]"
docker run -d -p 9092:9092 apache/kafka:3.8.0   # for the integration tests
pytest -v
```

Unit tests run against the in-memory bus. The integration tests run the whole pipeline against a real
broker and check compaction, offset commits, restart and state restore, and that a replayed transaction is
deduplicated. They're skipped when no broker is reachable; CI runs them against a Kafka service container.

The integration tests caught a real bug: aiokafka runs the rebalance callback concurrently with fetching, so
a restarted scorer could process records before their account state was restored. The scorer now holds a
lock during restore that batches wait on.

## Configuration

See `.env.example`. `KAFKA_BOOTSTRAP_SERVERS` defaults to `127.0.0.1:9092` rather than `localhost`: on
Windows, `localhost` tries IPv6 first, and each connection to a Docker-published port stalls for about 20
seconds before falling back.

## Limitations

- Synthetic data only. Real deployments need real labels, threshold tuning per segment, and a feedback
  loop from analyst decisions.
- Rule weights are hand-set; the Isolation Forest is trained on simulated normal traffic at startup.
- Processing is at-least-once with dedupe, not exactly-once (no Kafka transactions).
- The explainer's in-process dedupe doesn't survive restarts; alerts are keyed by transaction id so
  downstream consumers (and the API) collapse duplicates.

## License

MIT
