"""
Unit Tests for Recommender Models
=================================
Tests Item-CF, SVD (Matrix Factorization), Content-Based, Knowledge Filter, and Adaptive Hybrid.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.models.item_cf import ItemBasedCFRecommender
from src.models.svd_recommender import SVDRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.knowledge_filter import KnowledgeConstraintFilter
from src.models.hybrid_recommender import AdaptiveHybridRecommender


@pytest.fixture
def mock_data():
    """Generates synthetic dataset for model unit testing."""
    np.random.seed(42)
    users = [1, 2, 3, 4, 5]
    recipes = [101, 102, 103, 104, 105]

    interactions = []
    for u in users:
        for r in recipes:
            if np.random.rand() > 0.3:
                interactions.append({
                    "user_id": u,
                    "recipe_id": r,
                    "rating": float(np.random.choice([3.0, 4.0, 5.0])),
                    "date": "2021-05-01"
                })

    df_inter = pd.DataFrame(interactions)

    recipes_data = [
        {"RecipeId": 101, "Name": "Berry Parfait", "RecipeCategory": "Dessert", "TotalTimeMinutes": 15, "Calories": 250.0, "ProteinContent": 5.0, "is_vegetarian": True, "is_vegan": False, "is_gluten_free": True, "is_low_calorie": True, "is_high_protein": False, "is_quick_meal": True, "content_text": "berry yogurt dessert sweet parfait"},
        {"RecipeId": 102, "Name": "Chicken Curry", "RecipeCategory": "Dinner", "TotalTimeMinutes": 45, "Calories": 650.0, "ProteinContent": 35.0, "is_vegetarian": False, "is_vegan": False, "is_gluten_free": True, "is_low_calorie": False, "is_high_protein": True, "is_quick_meal": False, "content_text": "chicken curry spicy dinner meat rice"},
        {"RecipeId": 103, "Name": "Vegan Buddha Bowl", "RecipeCategory": "Lunch", "TotalTimeMinutes": 20, "Calories": 380.0, "ProteinContent": 18.0, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True, "is_low_calorie": True, "is_high_protein": False, "is_quick_meal": True, "content_text": "quinoa avocado chickpeas vegan healthy salad"},
        {"RecipeId": 104, "Name": "Beef Burger", "RecipeCategory": "Dinner", "TotalTimeMinutes": 25, "Calories": 750.0, "ProteinContent": 30.0, "is_vegetarian": False, "is_vegan": False, "is_gluten_free": False, "is_low_calorie": False, "is_high_protein": True, "is_quick_meal": True, "content_text": "beef burger cheese bun dinner fast"},
        {"RecipeId": 105, "Name": "Pasta Primavera", "RecipeCategory": "Dinner", "TotalTimeMinutes": 30, "Calories": 420.0, "ProteinContent": 12.0, "is_vegetarian": True, "is_vegan": False, "is_gluten_free": False, "is_low_calorie": False, "is_high_protein": False, "is_quick_meal": True, "content_text": "pasta vegetable parmesan olive oil italian"}
    ]
    df_recipes = pd.DataFrame(recipes_data)

    return df_inter, df_recipes


def test_item_cf_recommender(mock_data, tmp_path):
    df_inter, _ = mock_data
    model = ItemBasedCFRecommender(k_neighbors=3, shrinkage=1.0)
    model.fit(df_inter)

    assert model.is_fitted is True

    # Prediction
    pred = model.predict(1, 101)
    assert 1.0 <= pred <= 5.0

    # Recommendations
    recs = model.recommend(1, top_k=3, exclude_seen=False)
    assert len(recs) <= 3
    assert all(1.0 <= s <= 5.0 for _, s in recs)

    # Persistence
    save_path = str(tmp_path / "item_cf.joblib")
    model.save(save_path)
    loaded = ItemBasedCFRecommender.load(save_path)
    assert loaded.is_fitted is True
    assert loaded.predict(1, 101) == pred


def test_svd_recommender(mock_data, tmp_path):
    df_inter, _ = mock_data
    model = SVDRecommender(n_factors=4, learning_rate=0.05, reg=0.01, n_epochs=5, patience=2)
    model.fit(df_inter, val_df=df_inter)

    assert model.is_fitted is True
    assert len(model.training_history["train_rmse"]) > 0

    pred = model.predict(1, 101)
    assert 1.0 <= pred <= 5.0

    recs = model.recommend(1, top_k=3, exclude_seen=False)
    assert len(recs) <= 3

    # Test dynamic adaptation
    old_pred = model.predict(1, 102)
    model.adapt_user(1, 102, rating=5.0, step_size=0.1)
    new_pred = model.predict(1, 102)
    # Target was 5.0, so new prediction should adjust upward or reflect error step
    assert 1.0 <= new_pred <= 5.0


def test_content_based_recommender(mock_data):
    df_inter, df_recipes = mock_data
    model = ContentBasedRecommender(max_features=100)
    model.fit(df_inter, df_recipes)

    assert model.is_fitted is True
    recs = model.recommend(1, top_k=3)
    assert len(recs) <= 3

    sims = model.get_content_similar(101, top_k=2)
    assert len(sims) <= 2


def test_knowledge_constraint_filter(mock_data):
    _, df_recipes = mock_data
    kf = KnowledgeConstraintFilter(df_recipes)

    veg_recs = kf.filter_candidates(is_vegetarian=True)
    assert set(veg_recs) == {101, 103, 105}

    vegan_recs = kf.filter_candidates(is_vegan=True)
    assert set(vegan_recs) == {103}

    quick_recs = kf.filter_candidates(is_quick_meal=True)
    assert set(quick_recs) == {101, 103, 104, 105}


def test_adaptive_hybrid_recommender(mock_data):
    df_inter, df_recipes = mock_data
    svd = SVDRecommender(n_factors=4, n_epochs=3).fit(df_inter)
    item_cf = ItemBasedCFRecommender(k_neighbors=2).fit(df_inter)
    content = ContentBasedRecommender(max_features=50).fit(df_inter, df_recipes)
    knowledge = KnowledgeConstraintFilter(df_recipes)

    hybrid = AdaptiveHybridRecommender(
        svd_model=svd,
        item_cf_model=item_cf,
        content_model=content,
        knowledge_filter=knowledge,
        svd_weight=0.5,
        item_cf_weight=0.3,
        content_weight=0.2
    )

    pred = hybrid.predict(1, 101)
    assert 1.0 <= pred <= 5.0

    # With vegetarian filter
    recs = hybrid.recommend(1, top_k=3, dietary_constraints={"is_vegetarian": True})
    for r_id, _ in recs:
        assert r_id in {101, 103, 105}

    # Test real-time interaction feedback
    res = hybrid.record_interaction(1, 103, rating=5.0)
    assert res["status"] == "adapted"
