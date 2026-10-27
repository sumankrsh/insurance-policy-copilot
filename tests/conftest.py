from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.documents import load_chunks
from app.retrieval import Retriever
from ml.train_propensity import train


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """A small model trained into a temp dir, so tests don't depend on local files."""
    root = tmp_path_factory.mktemp("artifacts")
    report = train(
        n_customers=3000,
        seed=7,
        data_dir=root / "data",
        model_dir=root / "models",
        report_dir=None,
        log=lambda *_: None,
    )
    return root, report


@pytest.fixture
def settings(artifacts, tmp_path):
    root, _ = artifacts
    return replace(
        Settings(),
        llm_provider="extractive",
        data_dir=root / "data",
        model_dir=root / "models",
        audit_db=tmp_path / "audit.db",
    )


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def chunks():
    return load_chunks(Settings().docs_dir)


@pytest.fixture(scope="session")
def retriever(chunks):
    return Retriever(chunks)
