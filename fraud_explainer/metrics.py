"""Minimal in-process metrics with Prometheus text exposition (no extra dependency)."""

from __future__ import annotations

import threading
from collections import defaultdict

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self.histograms: dict[str, dict] = {}
        self.help: dict[str, str] = {}

    def inc(self, name: str, value: float = 1.0, help: str = "", **labels: str) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value
            if help:
                self.help.setdefault(name, help)

    def observe(self, name: str, seconds: float, help: str = "") -> None:
        with self._lock:
            h = self.histograms.setdefault(name, {"buckets": [0] * len(LATENCY_BUCKETS), "sum": 0.0, "count": 0})
            for i, bound in enumerate(LATENCY_BUCKETS):
                if seconds <= bound:
                    h["buckets"][i] += 1
            h["sum"] += seconds
            h["count"] += 1
            if help:
                self.help.setdefault(name, help)

    def get(self, name: str, **labels: str) -> float:
        return self.counters.get((name, tuple(sorted(labels.items()))), 0.0)

    def render(self) -> str:
        lines = []
        with self._lock:
            seen = set()
            for (name, labels), value in sorted(self.counters.items()):
                if name not in seen:
                    if name in self.help:
                        lines.append(f"# HELP {name} {self.help[name]}")
                    lines.append(f"# TYPE {name} counter")
                    seen.add(name)
                label_text = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{label_text}}} {value:g}" if label_text else f"{name} {value:g}")
            for name, h in sorted(self.histograms.items()):
                if name in self.help:
                    lines.append(f"# HELP {name} {self.help[name]}")
                lines.append(f"# TYPE {name} histogram")
                for bound, count in zip(LATENCY_BUCKETS, h["buckets"]):
                    lines.append(f'{name}_bucket{{le="{bound}"}} {count}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {h["count"]}')
                lines.append(f"{name}_sum {h['sum']:.6f}")
                lines.append(f"{name}_count {h['count']}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.histograms.clear()


METRICS = Metrics()
