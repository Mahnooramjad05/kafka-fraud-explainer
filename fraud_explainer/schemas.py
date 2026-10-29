"""Event schemas for every topic in the pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Action = Literal["approve", "review", "block"]


class Transaction(BaseModel):
    """A card transaction on `transactions.raw`, keyed by account_id."""

    transaction_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    timestamp: datetime
    amount: float = Field(gt=0)
    currency: str = "USD"
    merchant: str
    merchant_category: str
    city: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Features(BaseModel):
    """Per-account context computed by the scorer at the moment of the transaction."""

    amount_zscore: float = 0.0
    amount_ratio_to_mean: float = 1.0
    tx_count_10min: int = 0
    small_tx_count_10min: int = 0
    km_from_last: float = 0.0
    speed_kmh: float = 0.0
    new_category: bool = False
    night_hour: bool = False
    history_size: int = 0


class Signal(BaseModel):
    """A named, human-readable reason the scorer flagged a transaction."""

    name: str
    detail: str
    weight: float = Field(ge=0, le=1)


class ScoredTransaction(BaseModel):
    """A transaction with its risk score, on `transactions.scored`."""

    transaction: Transaction
    features: Features
    model_score: float = Field(ge=0, le=1, description="Isolation Forest anomaly score scaled to [0, 1].")
    risk: float = Field(ge=0, le=1)
    signals: list[Signal] = Field(default_factory=list)
    scored_at: datetime


class Explanation(BaseModel):
    """Structured LLM output for one flagged transaction."""

    transaction_id: str
    summary: str
    cited_signals: list[str] = Field(description="Signal names the explanation relies on.")
    recommended_action: Action
    confidence: float = Field(ge=0, le=1)


class Alert(BaseModel):
    """An analyst-facing alert on the `alerts` topic."""

    transaction_id: str
    account_id: str
    amount: float
    merchant: str
    city: str
    risk: float
    signals: list[Signal]
    explanation: Explanation
    faithful: bool = Field(description="True if the explanation only cites signals that actually fired.")
    unsupported_signals: list[str] = Field(default_factory=list)
    explainer: str
    created_at: datetime


class DeadLetter(BaseModel):
    """A message that could not be processed, on `transactions.dlq`."""

    source_topic: str
    stage: Literal["scorer", "explainer"]
    error: str
    attempts: int = 1
    key: Optional[str] = None
    payload: str = Field(description="The original message value, as text.")
    failed_at: datetime
