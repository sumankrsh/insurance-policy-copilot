"""Settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ml.decision import HIGH_RISK, MEDIUM_RISK

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = ROOT / "data" / "policy_documents"
    data_dir: Path = ROOT / "data" / "generated"
    model_dir: Path = ROOT / "models"
    audit_db: Path = ROOT / "var" / "audit.db"

    # "auto" picks anthropic if ANTHROPIC_API_KEY is set, else azure_openai if its
    # settings are present, else the offline extractive answerer.
    llm_provider: str = "auto"
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    llm_timeout_seconds: float = 60.0

    top_k: int = 4
    min_retrieval_score: float = 0.12

    high_risk_threshold: float = HIGH_RISK
    medium_risk_threshold: float = MEDIUM_RISK

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            docs_dir=Path(_env("DOCS_DIR", str(d.docs_dir))),
            data_dir=Path(_env("DATA_DIR", str(d.data_dir))),
            model_dir=Path(_env("MODEL_DIR", str(d.model_dir))),
            audit_db=Path(_env("AUDIT_DB_PATH", str(d.audit_db))),
            llm_provider=_env("LLM_PROVIDER", d.llm_provider).lower(),
            anthropic_model=_env("ANTHROPIC_MODEL", d.anthropic_model),
            anthropic_effort=_env("ANTHROPIC_EFFORT", d.anthropic_effort),
            azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=_env("AZURE_OPENAI_API_KEY"),
            azure_openai_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            azure_openai_api_version=_env("AZURE_OPENAI_API_VERSION", d.azure_openai_api_version),
            llm_timeout_seconds=float(_env("LLM_TIMEOUT_SECONDS", str(d.llm_timeout_seconds))),
            top_k=int(_env("RETRIEVAL_TOP_K", str(d.top_k))),
            min_retrieval_score=float(_env("MIN_RETRIEVAL_SCORE", str(d.min_retrieval_score))),
            high_risk_threshold=float(_env("HIGH_RISK_THRESHOLD", str(d.high_risk_threshold))),
            medium_risk_threshold=float(_env("MEDIUM_RISK_THRESHOLD", str(d.medium_risk_threshold))),
        )
