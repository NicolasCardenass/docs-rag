from fastapi.testclient import TestClient

from docs_rag import __version__


def test_health_returns_ok_and_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_openapi_schema_lists_health_endpoint(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "Docs RAG API"
    assert "/health" in schema["paths"]


def test_cors_allows_configured_frontend_origin(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
