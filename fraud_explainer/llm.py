"""
LLM explainers: turn a batch of flagged transactions into analyst-ready explanations.

The model gets the transaction, its computed features and the signals that
fired, and must return structured JSON per transaction. It is told to cite
only the signals provided; `check_faithfulness` verifies that afterwards.

Providers:
  mock       deterministic, offline (default; used by tests and the demo)
  openai     OPENAI_API_KEY, model from OPENAI_MODEL (default gpt-4o-mini)
  anthropic  ANTHROPIC_API_KEY, model from ANTHROPIC_MODEL (default claude-opus-5-5)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

from fraud_explainer.schemas import Explanation, ScoredTransaction

logger = logging.getLogger(__name__)


class ExplainerError(RuntimeError):
    """The batch could not be explained; the explainer service will retry it."""


SYSTEM_PROMPT = (
    "You are a fraud analyst assistant. For each flagged card transaction, write a short "
    "explanation an analyst can act on. Base it ONLY on the signals and features provided; "
    "do not invent reasons. In cited_signals, list only names that appear in that "
    "transaction's signals. Recommend 'block' for clear fraud patterns with high risk, "
    "'review' when a human should check, and 'approve' only if the signals are weak."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "cited_signals": {"type": "array", "items": {"type": "string"}},
                    "recommended_action": {"type": "string", "enum": ["approve", "review", "block"]},
                    "confidence": {"type": "number"},
                },
                "required": ["transaction_id", "summary", "cited_signals",
                             "recommended_action", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["explanations"],
    "additionalProperties": False,
}


def batch_prompt(batch: list[ScoredTransaction]) -> str:
    items = []
    for s in batch:
        tx = s.transaction
        items.append({
            "transaction_id": tx.transaction_id,
            "amount": tx.amount,
            "currency": tx.currency,
            "merchant": tx.merchant,
            "merchant_category": tx.merchant_category,
            "city": tx.city,
            "time_utc": tx.timestamp.isoformat(),
            "risk": s.risk,
            "features": s.features.model_dump(),
            "signals": [{"name": sig.name, "detail": sig.detail} for sig in s.signals],
        })
    return ("Explain each flagged transaction. Return one entry per transaction_id.\n\n"
            + json.dumps(items, indent=2))


def parse_explanations(data: dict, batch: list[ScoredTransaction]) -> list[Explanation]:
    """Map the model's JSON back onto the batch; any missing or malformed entry fails the batch."""
    by_id = {}
    for item in data.get("explanations", []):
        try:
            exp = Explanation.model_validate(item)
        except ValueError as exc:
            raise ExplainerError(f"malformed explanation: {exc}") from exc
        by_id[exp.transaction_id] = exp
    missing = [s.transaction.transaction_id for s in batch if s.transaction.transaction_id not in by_id]
    if missing:
        raise ExplainerError(f"no explanation returned for {missing}")
    return [by_id[s.transaction.transaction_id] for s in batch]


def check_faithfulness(scored: ScoredTransaction, explanation: Explanation) -> tuple[bool, list[str]]:
    """An explanation is faithful if it cites at least one signal and only signals that fired."""
    fired = {s.name for s in scored.signals}
    unsupported = sorted(set(explanation.cited_signals) - fired)
    return bool(explanation.cited_signals) and not unsupported, unsupported


class Explainer(ABC):
    name = "base"

    @abstractmethod
    async def explain(self, batch: list[ScoredTransaction]) -> list[Explanation]: ...

    async def aclose(self) -> None:
        pass


class MockExplainer(Explainer):
    """Deterministic template-based explanations, so the pipeline runs without an API key."""

    name = "mock"

    async def explain(self, batch: list[ScoredTransaction]) -> list[Explanation]:
        out = []
        for s in batch:
            tx = s.transaction
            reasons = "; ".join(sig.detail for sig in s.signals) or "no specific signal"
            action = "block" if s.risk >= 0.85 else "review"
            out.append(Explanation(
                transaction_id=tx.transaction_id,
                summary=(f"${tx.amount:,.2f} at {tx.merchant} ({tx.city}) flagged with risk "
                         f"{s.risk:.2f}: {reasons}."),
                cited_signals=[sig.name for sig in s.signals],
                recommended_action=action,
                confidence=round(s.risk, 2),
            ))
        return out


class AnthropicExplainer(Explainer):
    name = "anthropic"

    def __init__(self, model: str | None = None, client=None):
        import anthropic

        self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-opus-5-5"
        self.client = client or anthropic.AsyncAnthropic()

    async def explain(self, batch: list[ScoredTransaction]) -> list[Explanation]:
        import anthropic

        try:
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": batch_prompt(batch)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
                # On a safety decline, the API retries on a fallback model within the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            raise ExplainerError(f"Anthropic API error: {exc}") from exc
        if response.stop_reason == "refusal":
            raise ExplainerError("explanation request was declined")
        if response.stop_reason == "max_tokens":
            raise ExplainerError("explanation hit max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return parse_explanations(json.loads(text), batch)
        except json.JSONDecodeError as exc:
            raise ExplainerError(f"invalid JSON from model: {exc.msg}") from exc

    async def aclose(self) -> None:
        await self.client.close()


class OpenAIExplainer(Explainer):
    name = "openai"

    def __init__(self, model: str | None = None, client=None):
        import openai

        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.client = client or openai.AsyncOpenAI()

    async def explain(self, batch: list[ScoredTransaction]) -> list[Explanation]:
        import openai

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": batch_prompt(batch)}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "explanations", "schema": RESPONSE_SCHEMA, "strict": True}},
            )
        except openai.OpenAIError as exc:
            raise ExplainerError(f"OpenAI API error: {exc}") from exc
        message = response.choices[0].message
        if getattr(message, "refusal", None) or not message.content:
            raise ExplainerError("explanation request was declined or empty")
        try:
            return parse_explanations(json.loads(message.content), batch)
        except json.JSONDecodeError as exc:
            raise ExplainerError(f"invalid JSON from model: {exc.msg}") from exc

    async def aclose(self) -> None:
        await self.client.close()


def build_explainer(provider: str) -> Explainer:
    """Pick a provider; fall back to the mock when its API key isn't set."""
    provider = provider.lower()
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicExplainer()
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIExplainer()
    if provider not in ("mock", ""):
        logger.warning("No API key for LLM_PROVIDER=%s; using the offline mock explainer.", provider)
    return MockExplainer()
