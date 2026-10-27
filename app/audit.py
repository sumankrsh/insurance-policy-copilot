"""Audit log and human-review queue, both in one SQLite file.

The audit log is append-only and hash-chained: each record stores the SHA-256 of the
previous record, so editing or deleting a row breaks the chain and `verify()` reports
where. It stores redacted text only; raw user input never reaches it.

The review queue holds high-risk retention recommendations until a person approves
or rejects them. Every create/decide is also written to the audit log by the API.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash(prev_hash: str, ts: str, trace_id: str, event: str, payload: str) -> str:
    return hashlib.sha256(f"{prev_hash}|{ts}|{trace_id}|{event}|{payload}".encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self._conn = _connect(path)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                )"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS audit_trace ON audit_log(trace_id)")
            self._conn.commit()

    def record(self, trace_id: str, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            row = self._conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
            prev_hash = row["hash"] if row else GENESIS
            ts = _now()
            digest = _hash(prev_hash, ts, trace_id, event, body)
            cur = self._conn.execute(
                "INSERT INTO audit_log (ts, trace_id, event, payload, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, trace_id, event, body, prev_hash, digest),
            )
            self._conn.commit()
        return {"id": cur.lastrowid, "ts": ts, "trace_id": trace_id, "event": event, "hash": digest}

    def query(self, trace_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit_log"
        args: tuple = ()
        if trace_id:
            sql += " WHERE trace_id = ?"
            args = (trace_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*args, limit)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def verify(self) -> dict[str, Any]:
        """Recompute the hash chain. ok=False names the first record that doesn't match."""
        prev = GENESIS
        count = 0
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        for r in rows:
            expected = _hash(prev, r["ts"], r["trace_id"], r["event"], r["payload"])
            if r["prev_hash"] != prev or r["hash"] != expected:
                return {"ok": False, "records": len(rows), "first_bad_id": r["id"]}
            prev = r["hash"]
            count += 1
        return {"ok": True, "records": count, "first_bad_id": None}


class ReviewQueue:
    def __init__(self, path: Path):
        self._conn = _connect(path)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    probability REAL NOT NULL,
                    risk_tier TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    top_factors TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    reviewer TEXT,
                    note TEXT,
                    decided_at TEXT
                )"""
            )
            self._conn.commit()

    @staticmethod
    def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
        if r is None:
            return None
        d = dict(r)
        d["top_factors"] = json.loads(d["top_factors"])
        return d

    def pending_for_policy(self, policy_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM reviews WHERE policy_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
                (policy_id,),
            ).fetchone()
        return self._row(r)

    def create(self, *, trace_id: str, customer_id: str, policy_id: str, probability: float, risk_tier: str,
               recommended_action: str, top_factors: list[dict], model_version: str) -> dict[str, Any]:  # fmt: skip
        review_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                """INSERT INTO reviews (review_id, created_at, trace_id, customer_id, policy_id, probability,
                   risk_tier, recommended_action, top_factors, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (review_id, _now(), trace_id, customer_id, policy_id, probability, risk_tier,
                 recommended_action, json.dumps(top_factors), model_version),
            )  # fmt: skip
            self._conn.commit()
        return self.get(review_id)

    def get(self, review_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
        return self._row(r)

    def list(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM reviews", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        return [self._row(r) for r in rows]

    def decide(self, review_id: str, *, approve: bool, reviewer: str, note: str = "") -> dict[str, Any]:
        """Approve or reject a pending review. Raises KeyError if missing, ValueError if already decided."""
        with self._lock:
            r = self._conn.execute("SELECT status FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
            if r is None:
                raise KeyError(review_id)
            if r["status"] != "pending":
                raise ValueError(f"review {review_id} is already {r['status']}")
            self._conn.execute(
                "UPDATE reviews SET status = ?, reviewer = ?, note = ?, decided_at = ? WHERE review_id = ?",
                ("approved" if approve else "rejected", reviewer, note, _now(), review_id),
            )
            self._conn.commit()
        return self.get(review_id)
