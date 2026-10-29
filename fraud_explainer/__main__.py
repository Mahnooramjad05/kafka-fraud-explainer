"""
Command-line entry points:

    python -m fraud_explainer simulate  [--events N] [--rate EPS] [--fraud-rate R] [--seed S]
    python -m fraud_explainer scorer    [--metrics-port 8001]
    python -m fraud_explainer explainer [--metrics-port 8002]
    python -m fraud_explainer api       [--port 8000]

All read Kafka and pipeline settings from environment variables (see .env.example).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

import uvicorn

from fraud_explainer.api import create_app
from fraud_explainer.bus import build_bus
from fraud_explainer.config import Settings
from fraud_explainer.explainer import ExplainerService
from fraud_explainer.llm import build_explainer
from fraud_explainer.model import DetectionModel
from fraud_explainer.producer import publish
from fraud_explainer.scorer import ScorerService
from fraud_explainer.simulator import Simulator

logger = logging.getLogger("fraud_explainer")


def _stop_event() -> asyncio.Event:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # Windows: rely on KeyboardInterrupt
            pass
    return stop


async def _serve_metrics(port: int, role: str, stop: asyncio.Event) -> None:
    config = uvicorn.Config(create_app(role=role), host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await stop.wait()
    server.should_exit = True
    await task


async def _wait_for_kafka(bus, attempts: int = 30) -> None:
    for i in range(attempts):
        try:
            await bus.ensure_topics()
            return
        except Exception as exc:  # broker not up yet
            logger.info("waiting for Kafka (%s/%s): %s", i + 1, attempts, exc)
            await asyncio.sleep(2)
    raise RuntimeError("Kafka did not become available")


async def run_simulate(args, settings: Settings) -> None:
    bus = build_bus(settings)
    await _wait_for_kafka(bus)
    sim = Simulator(num_accounts=args.accounts, seed=args.seed, fraud_rate=args.fraud_rate)
    sent = await publish(bus, settings, sim, args.events, args.rate, args.malformed_every)
    frauds = sum(1 for e in sent if e.label)
    logger.info("published %d events (%d labeled fraud) to %s", len(sent), frauds, settings.topics.raw)


async def run_scorer(args, settings: Settings) -> None:
    bus = build_bus(settings)
    await _wait_for_kafka(bus)
    logger.info("training anomaly model on %d fraud-free simulated transactions", args.train_events)
    model = DetectionModel().fit(Simulator(seed=args.train_seed, fraud_rate=0).normal_history(args.train_events))
    stop = _stop_event()
    await asyncio.gather(ScorerService(bus, settings, model).run(stop),
                         _serve_metrics(args.metrics_port, "scorer", stop))


async def run_explainer(args, settings: Settings) -> None:
    bus = build_bus(settings)
    await _wait_for_kafka(bus)
    explainer = build_explainer(settings.llm_provider)
    logger.info("explaining with %s", explainer.name)
    stop = _stop_event()
    await asyncio.gather(ExplainerService(bus, settings, explainer).run(stop),
                         _serve_metrics(args.metrics_port, "explainer", stop))


def run_api(args, settings: Settings) -> None:
    bus = build_bus(settings)
    asyncio.run(_wait_for_kafka(bus))
    uvicorn.run(create_app(bus=bus, settings=settings), host="0.0.0.0", port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraud_explainer")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="publish simulated transactions")
    sim.add_argument("--events", type=int, default=5000)
    sim.add_argument("--rate", type=float, default=50.0, help="events per second (0 = unthrottled)")
    sim.add_argument("--accounts", type=int, default=200)
    sim.add_argument("--fraud-rate", type=float, default=0.03)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--malformed-every", type=int, default=None)

    scorer = sub.add_parser("scorer", help="run the scoring consumer")
    scorer.add_argument("--metrics-port", type=int, default=8001)
    scorer.add_argument("--train-events", type=int, default=6000)
    scorer.add_argument("--train-seed", type=int, default=1)

    explainer = sub.add_parser("explainer", help="run the LLM explainer consumer")
    explainer.add_argument("--metrics-port", type=int, default=8002)

    api = sub.add_parser("api", help="serve the alerts API")
    api.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aiokafka").setLevel(logging.WARNING)
    settings = Settings.from_env()

    if args.command == "api":
        run_api(args, settings)
        return
    runner = {"simulate": run_simulate, "scorer": run_scorer, "explainer": run_explainer}[args.command]
    try:
        asyncio.run(runner(args, settings))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
