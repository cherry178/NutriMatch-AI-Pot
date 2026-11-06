"""
Redesign Verification Tests for NutriMatch AI Pot
=================================================
Verifies that the redesigned NutriMatch AI Pot frontend:
1. Is served properly on GET /
2. Preserves brand identity: NutriMatch AI Pot, tagline, formal title
3. Has the 5 main navigation destinations:
   - Discover Recipes
   - Saved Recipes
   - My Taste Profile
   - Model Evaluation
   - Research & Diagnostics
4. Has simplified header elements: logo, search, saved count, profile, online status
5. Has real filter elements (vegetarian, vegan, gluten-free, quick, low-cal, protein)
6. Has latent space and ingredient graph canvases
7. Connects with live API endpoints
"""

import os
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_frontend_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Brand identity assertions
    assert "NutriMatch AI Pot" in html
    assert "Your Taste. Your Recipes. Your AI." in html
    assert "Adaptive Personalized Recipe Recommendation Engine" in html

    # Main 5 navigation destinations
    assert 'id="nav-item-discover"' in html
    assert 'id="nav-item-saved"' in html
    assert 'id="nav-item-profile"' in html
    assert 'id="nav-item-benchmarks"' in html
    assert 'id="nav-item-research"' in html

    # Clean header assertions
    assert 'id="header-search-input"' in html
    assert 'id="saved-counter-badge"' in html
    assert 'id="user-menu-btn"' in html
    assert 'id="sidebar-toggle-btn"' in html

    # Filter chips
    assert 'id="filter-veg"' in html
    assert 'id="filter-vegan"' in html
    assert 'id="filter-gf"' in html
    assert 'id="filter-quick"' in html
    assert 'id="filter-low-cal"' in html
    assert 'id="filter-protein"' in html
    assert 'id="filter-max-time"' in html

    # Canvases for Latent Space and Ingredient Graph
    assert 'id="latent-canvas"' in html
    assert 'id="ingredient-canvas"' in html

    # Modals
    assert 'id="new-profile-modal"' in html
    assert 'id="team-modal"' in html

def test_api_endpoints_integrity(client):
    # Recommendations
    res_rec = client.post("/api/v1/recommend", json={"user_id": 1535, "top_k": 6, "model_type": "hybrid"})
    assert res_rec.status_code == 200
    recs = res_rec.json()["recommendations"]
    assert len(recs) <= 6
    assert all("name" in r and "predicted_score" in r for r in recs)

    # User Profile
    res_prof = client.get("/api/v1/user/profile/1535")
    assert res_prof.status_code == 200
    prof = res_prof.json()
    assert "vegetarian_affinity" in prof["metrics"]

    # Benchmark Data
    res_bench = client.get("/api/v1/benchmark")
    assert res_bench.status_code == 200
    bench = res_bench.json()
    assert "Adaptive Hybrid Recommender" in bench["metrics"]

    # Diagnostics Data
    res_diag = client.get("/api/v1/diagnostics")
    assert res_diag.status_code == 200
    diag = res_diag.json()
    assert "val_rmse" in diag["history"]

    # Latent Space
    res_ls = client.get("/api/v1/latent-space?user_id=1535")
    assert res_ls.status_code == 200
    assert len(res_ls.json()["recipes"]) > 0

    # Ingredient Graph
    res_ig = client.get("/api/v1/ingredient-graph")
    assert res_ig.status_code == 200
    assert len(res_ig.json()["nodes"]) == 24


def test_recipe_instructions_and_ingredients_endpoints(client):
    """Verify at least 3 distinct recipe IDs return matching ingredients and cooking instructions."""
    test_recipe_ids = [426090, 95021, 15826]
    for rid in test_recipe_ids:
        res = client.get(f"/api/v1/recipes/{rid}")
        assert res.status_code == 200, f"Failed for recipe {rid}"
        data = res.json()
        assert data["recipe_id"] == rid
        assert "name" in data and len(data["name"]) > 0
        assert "ingredients" in data and isinstance(data["ingredients"], list)
        assert len(data["ingredients"]) > 0
        assert "instructions" in data and isinstance(data["instructions"], list)
        assert len(data["instructions"]) > 0
        assert "steps" in data and isinstance(data["steps"], list)
        assert data["steps"] == data["instructions"]
        # Check that steps are sequential strings
        assert all(isinstance(step, str) and len(step.strip()) > 0 for step in data["instructions"])


def test_recommendations_contain_instructions(client):
    """Verify recommendation feed includes instructions and steps for all items."""
    res = client.post("/api/v1/recommend", json={"user_id": 1535, "top_k": 5, "model_type": "hybrid"})
    assert res.status_code == 200
    recs = res.json()["recommendations"]
    assert len(recs) > 0
    for r in recs:
        assert "ingredients" in r and isinstance(r["ingredients"], list)
        assert "instructions" in r and isinstance(r["instructions"], list)
        assert "steps" in r and isinstance(r["steps"], list)
        assert len(r["instructions"]) > 0


def test_frontend_markup_for_instructions(client):
    """Verify frontend HTML contains the step-by-step cooking instructions markup and fallback."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "Step-by-Step Cooking Instructions" in html
    assert "Cooking instructions are not available for this recipe." in html
    assert "recipe-cards-container" in html
    assert "items-start" in html

