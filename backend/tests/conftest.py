from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docs_rag.config import Settings
from docs_rag.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings isolated from the environment, storing data in a temp dir."""
    return Settings(_env_file=None, environment="test", data_dir=tmp_path / "data")  # type: ignore[call-arg]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
