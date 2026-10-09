from fastapi.testclient import TestClient


def test_health_reports_ok_and_seeded_stock_count(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "stock_items": 150}
