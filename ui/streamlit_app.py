"""Streamlit front end for the policy copilot API.

streamlit run ui/streamlit_app.py      # expects the API at API_BASE_URL (default http://localhost:8000)
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Insurance Policy Copilot", layout="wide")


def call(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API}{path}", timeout=90, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Can't reach the API at {API}: {exc}")
        return None
    if r.status_code >= 400:
        detail = (
            r.json().get("detail", r.text)
            if r.headers.get("content-type", "").startswith("application/json")
            else r.text
        )
        st.error(f"{r.status_code}: {detail}")
        return None
    return r.json()


health = call("GET", "/health")
st.title("Insurance Policy Copilot")
st.caption(
    "Policy Q&A with citations, surrender-risk scoring with human review, and an audit log. All data is synthetic."
)
if health:
    st.caption(
        f"Answers: `{health['answer_provider']}` · Propensity model: `{health['propensity_model'] or 'not trained'}` "
        f"· {health['document_chunks']} document sections indexed"
    )

ask_tab, risk_tab, review_tab, audit_tab = st.tabs(["Ask the documents", "Retention risk", "Review queue", "Audit log"])

with ask_tab:
    examples = [
        "What is the surrender charge in year 2 of the term life policy?",
        "How long is the free-look period?",
        "What is the waiting period for pre-existing diseases?",
        "What is the management fee for the Equity Growth Fund?",
    ]
    with st.form("ask"):
        question = st.text_input("Question", placeholder=examples[0])
        submitted = st.form_submit_button("Ask")
    st.caption("Try: " + " · ".join(examples[1:]))
    if submitted and question.strip():
        result = call("POST", "/ask", json={"question": question})
        if result:
            st.markdown(result["answer"])
            for c in result["citations"]:
                with st.expander(f"[{c['n']}] {c['document']}, page {c['page']}, {c['section']}"):
                    st.text(c["quote"] or "")
            notes = ", ".join(result["guardrails"]) or "none"
            st.caption(f"Answered by `{result['provider']}` · guardrails: {notes} · trace `{result['trace_id']}`")

with risk_tab:
    with st.form("risk"):
        customer_id = st.text_input("Customer ID", value="C000001", help="Synthetic IDs run from C000001 to C020000")
        scored = st.form_submit_button("Score")
    if scored:
        result = call("POST", f"/customers/{customer_id.strip()}/propensity")
        if result and not result["policies"]:
            st.info("This customer has no active policies.")
        elif result:
            d = result["decision"]
            cols = st.columns(3)
            cols[0].metric("Highest 90-day surrender probability", f"{d['probability']:.0%}")
            cols[1].metric("Risk tier", d["risk_tier"])
            cols[2].caption("Recommended action")
            cols[2].markdown(f"**{d['recommended_action'].replace('_', ' ')}**")
            if d["status"] == "pending_human_review":
                st.warning(f"Held for human review before any outreach (review `{d['review_id']}`).")
            rows = [
                {
                    "policy": p["policy_id"],
                    "type": p["policy_type"],
                    "probability": f"{p['probability']:.0%}",
                    "tier": p["risk_tier"],
                    "top factors": "; ".join(
                        f"{f['description']} = {f['value']:g} (+{f['effect'] * 100:.0f} pts)" for f in p["top_factors"]
                    ),
                }
                for p in result["policies"]
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(
                f"Model `{result['model_version']}` · factor effect = how much the score drops when that "
                f"feature is set to its typical value · trace `{result['trace_id']}`"
            )

with review_tab:
    reviewer = st.text_input("Your name (recorded in the audit log)", value="")
    pending = call("GET", "/reviews", params={"status": "pending"}) or []
    if not pending:
        st.info("No recommendations waiting for review. Score a high-risk customer in the Retention tab to add one.")
    for r in pending:
        with st.container(border=True):
            st.markdown(
                f"**{r['customer_id']}** · policy {r['policy_id']} · {r['probability']:.0%} "
                f"({r['risk_tier']}) · recommended: {r['recommended_action'].replace('_', ' ')}"
            )
            st.caption("; ".join(f"{f['description']} = {f['value']:g}" for f in r["top_factors"]))
            note = st.text_input("Note", key=f"note-{r['review_id']}")
            left, right, _ = st.columns([1, 1, 4])
            for col, approve, label in ((left, True, "Approve"), (right, False, "Reject")):
                if col.button(label, key=f"{label}-{r['review_id']}", disabled=not reviewer.strip()):
                    body = {"approve": approve, "reviewer": reviewer.strip(), "note": note}
                    if call("POST", f"/reviews/{r['review_id']}", json=body):
                        st.rerun()
    decided = [r for r in (call("GET", "/reviews") or []) if r["status"] != "pending"]
    if decided:
        st.subheader("Decided")
        st.dataframe(
            pd.DataFrame(decided)[
                ["review_id", "customer_id", "policy_id", "probability", "status", "reviewer", "note"]
            ],
            hide_index=True,
            width="stretch",
        )

with audit_tab:
    if st.button("Verify hash chain"):
        v = call("GET", "/audit/verify")
        if v and v["ok"]:
            st.success(f"Chain intact: {v['records']} records.")
        elif v:
            st.error(f"Chain broken at record {v['first_bad_id']}.")
    records = call("GET", "/audit", params={"limit": 50}) or []
    if records:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "time": r["ts"],
                        "event": r["event"],
                        "trace": r["trace_id"][:8],
                        "payload": r["payload"],
                    }
                    for r in records
                ]
            ),
            hide_index=True,
            width="stretch",
        )
