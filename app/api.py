"""FastAPI app. Use `create_app(settings)`; app/main.py builds the default instance."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from app import guardrails
from app.answer import PolicyQA, build_answerer
from app.audit import AuditLog, ReviewQueue
from app.config import Settings
from app.documents import load_chunks
from app.propensity import ModelNotTrained, PropensityService
from app.retrieval import Retriever

log = logging.getLogger(__name__)

CUSTOMER_ID = r"^C\d{6}$"
POLICY_ID = r"^P\d{8}$"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, examples=["What is the surrender charge in year 2?"])


class CitationOut(BaseModel):
    n: int
    document: str
    source_file: str
    page: int
    section: str
    quote: str | None = None


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    found: bool
    citations: list[CitationOut]
    provider: str
    guardrails: list[str]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    customer_id: str | None = Field(default=None, pattern=CUSTOMER_ID)
    policy_id: str | None = Field(default=None, pattern=POLICY_ID)


class ChatResponse(BaseModel):
    trace_id: str
    route: Literal["policy_question", "propensity", "surrender_value", "policy_summary", "blocked"]
    answer: str
    citations: list[CitationOut] = []
    data: dict[str, Any] | None = None
    guardrails: list[str] = []


class ReviewDecision(BaseModel):
    approve: bool
    reviewer: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=1000)


def _trace_id() -> str:
    return uuid.uuid4().hex


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    retriever = Retriever(load_chunks(settings.docs_dir))
    qa = PolicyQA(
        retriever,
        build_answerer(settings, retriever),
        top_k=settings.top_k,
        min_score=settings.min_retrieval_score,
    )
    propensity = PropensityService(
        settings.model_dir, settings.data_dir, settings.high_risk_threshold, settings.medium_risk_threshold
    )
    audit = AuditLog(settings.audit_db)
    reviews = ReviewQueue(settings.audit_db)

    app = FastAPI(
        title="Insurance Policy Copilot",
        description="Policy Q&A with citations, surrender-propensity scoring with human review, and an audit log. "
        "All data is synthetic.",
        version="0.2.0",
    )
    app.state.settings = settings
    app.state.audit = audit
    app.state.reviews = reviews

    # --- shared logic -------------------------------------------------------------------

    def answer_question(question: str, trace_id: str, event: str = "ask") -> tuple[dict[str, Any], list[str]]:
        start = time.perf_counter()
        check = guardrails.check_input(question)
        audit_payload: dict[str, Any] = {
            "question_redacted": check.text,
            "pii_types": sorted({e.entity for e in check.entities}),
            "input_flags": check.flags,
        }
        if check.blocked:
            audit.record(trace_id, event, {**audit_payload, "outcome": "blocked"})
            return {
                "answer": guardrails.INJECTION_REFUSAL,
                "found": False,
                "citations": [],
                "provider": "guardrail",
            }, check.flags

        answer, hits = qa.ask(check.text)
        text, output_flags = guardrails.check_output(answer.text, advice_request=check.advice_request)
        citations = [
            CitationOut(**{k: v for k, v in asdict(c).items() if k != "chunk_id"}).model_dump()
            for c in answer.citations
        ]
        audit.record(
            trace_id,
            event,
            {
                **audit_payload,
                "retrieved": [{"chunk_id": h.chunk.chunk_id, "score": h.score} for h in hits],
                "provider": answer.provider,
                "fallback_reason": answer.fallback_reason,
                "found": answer.found,
                "answer": text,
                "cited_chunks": [c.chunk_id for c in answer.citations],
                "output_flags": output_flags,
                "usage": answer.usage,
                "latency_ms": round((time.perf_counter() - start) * 1000),
            },
        )
        # Clients see the category ("llm_error"); the full reason stays in the audit log.
        fallback = [f"fallback: {answer.fallback_reason.split(':')[0]}"] if answer.fallback_reason else []
        flags = check.flags + output_flags + fallback
        return {"answer": text, "found": answer.found, "citations": citations, "provider": answer.provider}, flags

    def score_customer(customer_id: str, trace_id: str) -> dict[str, Any]:
        try:
            result = propensity.score_customer(customer_id)
        except ModelNotTrained as exc:
            raise HTTPException(503, str(exc)) from exc
        if result is None:
            raise HTTPException(404, f"customer {customer_id} not found")

        decision = result["decision"]
        if decision and decision["needs_human_review"]:
            top = result["policies"][0]
            review = reviews.pending_for_policy(decision["policy_id"])
            if review is None:
                review = reviews.create(
                    trace_id=trace_id,
                    customer_id=customer_id,
                    policy_id=decision["policy_id"],
                    probability=decision["probability"],
                    risk_tier=decision["risk_tier"],
                    recommended_action=decision["recommended_action"],
                    top_factors=top["top_factors"],
                    model_version=result["model_version"],
                )
                audit.record(trace_id, "review_created", {"review_id": review["review_id"], **decision})
            decision["status"] = "pending_human_review"
            decision["review_id"] = review["review_id"]
        elif decision:
            decision["status"] = "no_action" if decision["risk_tier"] == "low" else "recommended"
            decision["review_id"] = None

        audit.record(
            trace_id,
            "propensity_scored",
            {
                "customer_id": customer_id,
                "model_version": result["model_version"],
                "scores": [{k: p[k] for k in ("policy_id", "probability", "risk_tier")} for p in result["policies"]],
                "decision": decision,
            },
        )
        return result

    def lookup(kind: Literal["policy_summary", "surrender_value"], policy_id: str, trace_id: str) -> dict[str, Any]:
        fn = propensity.policy_summary if kind == "policy_summary" else propensity.surrender_value
        try:
            result = fn(policy_id)
        except ModelNotTrained as exc:
            raise HTTPException(503, str(exc)) from exc
        if result is None:
            raise HTTPException(404, f"policy {policy_id} not found")
        audit.record(trace_id, kind, {"policy_id": policy_id})
        return result

    # --- endpoints ----------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "answer_provider": qa.provider,
            "document_chunks": len(retriever.chunks),
            "propensity_model": propensity.model_version if propensity.available else None,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        """Answer a question from the policy documents, with citations."""
        trace_id = _trace_id()
        result, flags = answer_question(req.question, trace_id)
        return AskResponse(trace_id=trace_id, guardrails=flags, **result)

    @app.post("/customers/{customer_id}/propensity")
    def propensity_endpoint(customer_id: str = Path(pattern=CUSTOMER_ID)) -> dict[str, Any]:
        """Score a customer's active policies. High-risk recommendations go to the review queue."""
        trace_id = _trace_id()
        return {"trace_id": trace_id, **score_customer(customer_id, trace_id)}

    @app.get("/policies/{policy_id}")
    def policy_summary(policy_id: str = Path(pattern=POLICY_ID)) -> dict[str, Any]:
        trace_id = _trace_id()
        return {"trace_id": trace_id, **lookup("policy_summary", policy_id, trace_id)}

    @app.get("/policies/{policy_id}/surrender-value")
    def surrender_value(policy_id: str = Path(pattern=POLICY_ID)) -> dict[str, Any]:
        trace_id = _trace_id()
        return {"trace_id": trace_id, **lookup("surrender_value", policy_id, trace_id)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        """One entry point for the UI. Routes by simple rules (see route_message)."""
        trace_id = _trace_id()
        check = guardrails.check_input(req.message)
        if check.blocked:
            audit.record(trace_id, "chat", {"message_redacted": check.text, "outcome": "blocked"})
            return ChatResponse(trace_id=trace_id, route="blocked", answer=guardrails.INJECTION_REFUSAL,
                                guardrails=check.flags)  # fmt: skip

        ids = {e.entity: e.value for e in check.entities}
        customer_id = req.customer_id or ids.get("CUSTOMER_ID")
        policy_id = req.policy_id or ids.get("POLICY_ID")
        route = route_message(req.message, customer_id, policy_id)
        audit.record(trace_id, "chat_routed", {"message_redacted": check.text, "route": route})

        if route == "propensity":
            result = score_customer(customer_id, trace_id)
            d = result["decision"]
            if d is None:
                text = f"Customer {customer_id} has no active policies to score."
            else:
                text = (
                    f"Highest surrender risk: policy {d['policy_id']} at {d['probability']:.0%} ({d['risk_tier']} risk). "
                    f"Recommended action: {d['recommended_action'].replace('_', ' ')}."
                )
                if d["status"] == "pending_human_review":
                    text += f" This is held for human review (review {d['review_id']}) before any outreach."
            return ChatResponse(trace_id=trace_id, route=route, answer=text, data=result, guardrails=check.flags)
        if route in ("surrender_value", "policy_summary"):
            result = lookup(route, policy_id, trace_id)
            if route == "surrender_value":
                text = (
                    f"Indicative surrender value of {policy_id}: {result['indicative_surrender_value']:,.2f} "
                    f"({result['surrender_charge_pct']}% surrender charge). {result['note']}"
                )
            else:
                text = (
                    f"{policy_id}: {result['policy_type']}, {result['status']}, premium "
                    f"{result['premium_amount']:,.2f} {result['premium_frequency'].lower()}, "
                    f"sum assured {result['sum_assured']:,.2f}."
                )
            return ChatResponse(trace_id=trace_id, route=route, answer=text, data=result, guardrails=check.flags)

        result, flags = answer_question(req.message, trace_id, event="chat_answer")
        return ChatResponse(trace_id=trace_id, route="policy_question", answer=result["answer"],
                            citations=result["citations"], data={"provider": result["provider"]},
                            guardrails=flags)  # fmt: skip

    @app.get("/reviews")
    def list_reviews(status: Literal["pending", "approved", "rejected"] | None = None) -> list[dict[str, Any]]:
        return reviews.list(status)

    @app.post("/reviews/{review_id}")
    def decide_review(review_id: str, body: ReviewDecision) -> dict[str, Any]:
        """Approve or reject a held recommendation."""
        try:
            review = reviews.decide(review_id, approve=body.approve, reviewer=body.reviewer, note=body.note)
        except KeyError as exc:
            raise HTTPException(404, f"review {review_id} not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        redacted_note, _ = guardrails.redact(body.note)
        audit.record(
            review["trace_id"],
            "review_decided",
            {"review_id": review_id, "status": review["status"], "reviewer": body.reviewer, "note": redacted_note},
        )
        return review

    @app.get("/audit")
    def audit_records(
        trace_id: str | None = None, limit: int = Query(default=50, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        return audit.query(trace_id, limit)

    @app.get("/audit/verify")
    def audit_verify() -> dict[str, Any]:
        """Recompute the audit log's hash chain."""
        return audit.verify()

    return app


PROPENSITY_WORDS = re.compile(r"\b(propensity|risk|likely to (surrender|lapse|leave|cancel)|churn|retention)\b", re.I)
SURRENDER_VALUE_WORDS = re.compile(r"\b(surrender value|how much (would|will|do) (i|they|we) get|payout)\b", re.I)
SUMMARY_WORDS = re.compile(r"\b(summary|summari[sz]e|details|status|overview)\b", re.I)


def route_message(message: str, customer_id: str | None, policy_id: str | None) -> str:
    """Deterministic routing. Tools need an ID; everything else is a document question."""
    if customer_id and PROPENSITY_WORDS.search(message):
        return "propensity"
    if policy_id and SURRENDER_VALUE_WORDS.search(message):
        return "surrender_value"
    if policy_id and SUMMARY_WORDS.search(message):
        return "policy_summary"
    return "policy_question"
