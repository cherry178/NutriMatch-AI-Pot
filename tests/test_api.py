"""
Integration Tests for FastAPI Endpoints
=======================================
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    # Use TestClient with context manager to trigger startup events
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_recommendation_endpoint(client):
    payload = {
        "user_id": 2046,
        "top_k": 5,
        "model_type": "hybrid",
        "dietary_constraints": {
            "is_vegetarian": True,
            "max_minutes": 60
        }
    }
    response = client.post("/api/v1/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 2046
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5
    for item in data["recommendations"]:
        assert item["is_vegetarian"] is True
        assert item["total_time_minutes"] <= 60


def test_feedback_endpoint(client):
    payload = {
        "user_id": 2046,
        "recipe_id": 4684,
        "rating": 5.0,
        "feedback_type": "rating"
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["user_id"] == 2046


def test_benchmark_endpoint(client):
    response = client.get("/api/v1/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model" in data
    assert "metrics" in data


def test_diagnostics_endpoint(client):
    response = client.get("/api/v1/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert "train_rmse" in data["history"]
    assert "val_rmse" in data["history"]


def test_search_endpoint(client):
    response = client.get("/api/v1/recipes/search?query=curry&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 5


def test_users_and_profile_endpoints(client):
    # Test users list
    res_users = client.get("/api/v1/users")
    assert res_users.status_code == 200
    users = res_users.json()
    assert len(users) >= 4
    assert any(u["user_id"] == 1535 for u in users)

    # Test user profile for user 1535
    res_prof = client.get("/api/v1/user/profile/1535")
    assert res_prof.status_code == 200
    prof = res_prof.json()
    assert prof["user_id"] == 1535
    assert "metrics" in prof
    assert "vegetarian_affinity" in prof["metrics"]
    assert "latent_weights" in prof


def test_visualization_and_export_endpoints(client):
    # Latent space
    res_ls = client.get("/api/v1/latent-space?user_id=1535")
    assert res_ls.status_code == 200
    ls_data = res_ls.json()
    assert "recipes" in ls_data
    assert "user_coordinate" in ls_data
    assert len(ls_data["recipes"]) > 0

    # Ingredient graph
    res_ig = client.get("/api/v1/ingredient-graph")
    assert res_ig.status_code == 200
    ig_data = res_ig.json()
    assert "nodes" in ig_data
    assert "links" in ig_data
    assert len(ig_data["nodes"]) > 0

    # Export logs
    res_exp = client.get("/api/v1/export-logs?user_id=1535")
    assert res_exp.status_code == 200
    exp_data = res_exp.json()
    assert "user_profile" in exp_data
    assert "model_parameters" in exp_data

