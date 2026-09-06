"""Request and response contracts for the scoring service."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

V_FIELDS = [f"V{i}" for i in range(1, 29)]


class _TransactionBase(BaseModel):
    """
    One transaction to score.

    Validation is strict on purpose. A scoring service that silently accepts a
    missing feature and imputes a zero will return a confident number for an
    input it never really saw, and nothing downstream will know. Every field is
    required and out-of-range values are rejected rather than clipped.

    The bounds on V1-V28 are wide (±200 against an observed range of about ±70)
    because they are a sanity check, not a business rule: they catch a unit
    error or a corrupted payload, and they do not reject a genuinely extreme
    transaction, which is exactly the kind this model exists to find.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(
        default=None, max_length=64,
        description="Echoed back in the response so a decision can be traced.",
    )
    time: float = Field(
        ge=0, le=60 * 60 * 24 * 366,
        description="Seconds since the start of the observation window.",
    )
    amount: float = Field(ge=0, le=1_000_000, description="Transaction amount.")

    @field_validator("amount", "time")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be a finite number")
        return v


#: The 28 anonymised components are generated rather than typed out: one
#: definition cannot drift from another, and the bound is stated once.
_Component = Annotated[float, Field(ge=-200.0, le=200.0)]

Transaction = create_model(
    "Transaction",
    __base__=_TransactionBase,
    **{
        name: (
            _Component,
            Field(..., description=f"Anonymised principal component {name}."),
        )
        for name in V_FIELDS
    },
)
Transaction.__doc__ = _TransactionBase.__doc__


class ReasonOut(BaseModel):
    feature: str
    contribution: float
    direction: str
    explanation: str


class Decision(BaseModel):
    """What the service decided, and enough context to defend the decision."""

    transaction_id: str | None
    score: float = Field(description="Estimated probability that this is fraud.")
    decision: str = Field(description="'review' or 'approve'.")
    threshold: float
    model_version: str
    reasons: list[ReasonOut] = Field(default_factory=list)
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)
    explain: bool = False


class BatchResponse(BaseModel):
    decisions: list[Decision]
    count: int
    flagged: int
    latency_ms: float


class Health(BaseModel):
    status: str
    model_version: str
    model_name: str
    threshold: float
    features: int
    trained_rows: int
    explainer_ready: bool
