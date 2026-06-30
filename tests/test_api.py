import json
from app import app


def test_health():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("status") == "ok"


def test_predict_api():
    client = app.test_client()
    payload = {
        "gender": "Female",
        "age": 29,
        "quantity": 1,
        "price_per_unit": 120,
        "date": "2023-08-15",
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert "product_category" in data
