"""
Adaptive Hybrid Recommendation Engine
======================================
Combines Collaborative Filtering (SVD Matrix Factorization and Item-Based CF),
Content-Based TF-IDF matching, and Knowledge-Based dietary constraint filtering
into a unified production recommendation engine.
Features real-time online adaptation: incoming user ratings immediately update
latent palate factors and taste centroids, re-ranking candidate recipes on the fly.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from src.models.base import BaseRecommender
from src.models.svd_recommender import SVDRecommender
from src.models.item_cf import ItemBasedCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.knowledge_filter import KnowledgeConstraintFilter

logger = logging.getLogger(__name__)


class AdaptiveHybridRecommender(BaseRecommender):
    """
    Industry-Grade Adaptive Hybrid Recipe Recommendation Engine.
    """

    def __init__(
        self,
        svd_model: Optional[SVDRecommender] = None,
        item_cf_model: Optional[ItemBasedCFRecommender] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        knowledge_filter: Optional[KnowledgeConstraintFilter] = None,
        svd_weight: float = 0.50,
        item_cf_weight: float = 0.30,
        content_weight: float = 0.20,
        adaptive_step_size: float = 0.05
    ):
        super().__init__(name="Adaptive Hybrid Recommender")
        self.svd = svd_model
        self.item_cf = item_cf_model
        self.content = content_model
        self.knowledge = knowledge_filter

        self.svd_weight = svd_weight
        self.item_cf_weight = item_cf_weight
        self.content_weight = content_weight
        self.adaptive_step_size = adaptive_step_size

        self.user_session_feedback: Dict[int, List[Dict[str, Any]]] = {}

    def fit(self, train_df: pd.DataFrame, recipes_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "AdaptiveHybridRecommender":
        """Fit all constituent models if not already fitted."""
        logger.info(f"Fitting {self.name} constituent models...")

        if self.svd is None or not self.svd.is_fitted:
            self.svd = SVDRecommender(n_factors=32, learning_rate=0.01, reg=0.05)
            self.svd.fit(train_df, val_df=val_df)

        if self.item_cf is None or not self.item_cf.is_fitted:
            self.item_cf = ItemBasedCFRecommender(k_neighbors=30, shrinkage=10.0)
            self.item_cf.fit(train_df)

        if self.content is None or not self.content.is_fitted:
            self.content = ContentBasedRecommender(max_features=4000)
            self.content.fit(train_df, recipes_df)

        if self.knowledge is None:
            self.knowledge = KnowledgeConstraintFilter(recipes_df)

        self.is_fitted = True
        logger.info(f"{self.name} successfully fitted and ready for inference.")
        return self

    def predict(self, user_id: int, recipe_id: int) -> float:
        """Predict weighted composite score."""
        svd_score = self.svd.predict(user_id, recipe_id) if self.svd else 3.5
        cf_score = self.item_cf.predict(user_id, recipe_id) if self.item_cf else 3.5
        cb_score = self.content.predict(user_id, recipe_id) if self.content else 3.5

        final_score = (
            self.svd_weight * svd_score +
            self.item_cf_weight * cf_score +
            self.content_weight * cb_score
        )
        return float(np.clip(final_score, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_ids: Optional[List[int]] = None,
        dietary_constraints: Optional[Dict[str, Any]] = None,
        offset: int = 0
    ) -> List[Tuple[int, float]]:
        """
        Generate adaptive hybrid recommendations:
        1. Apply Knowledge filter for dietary constraints (cascade stage).
        2. Score remaining candidates using weighted SVD + Item-CF + Content-Based predictions.
        3. Rank and return top-k recipes with pagination offset.
        """
        # Step 1: Filter candidates via knowledge rules
        if self.knowledge and dietary_constraints:
            valid_candidates = self.knowledge.filter_candidates(
                candidate_ids=candidate_ids,
                is_vegetarian=dietary_constraints.get("is_vegetarian", False),
                is_vegan=dietary_constraints.get("is_vegan", False),
                is_gluten_free=dietary_constraints.get("is_gluten_free", False),
                is_low_calorie=dietary_constraints.get("is_low_calorie", False),
                is_high_protein=dietary_constraints.get("is_high_protein", False),
                is_quick_meal=dietary_constraints.get("is_quick_meal", False),
                max_minutes=dietary_constraints.get("max_minutes", None),
                max_calories=dietary_constraints.get("max_calories", None),
                min_protein=dietary_constraints.get("min_protein", None),
                category=dietary_constraints.get("category", None),
                exclude_ingredients=dietary_constraints.get("exclude_ingredients", None),
                include_ingredients=dietary_constraints.get("include_ingredients", None)
            )
        else:
            if candidate_ids is not None:
                valid_candidates = candidate_ids
            elif self.svd:
                valid_candidates = list(self.svd.recipe_to_idx.keys())
            else:
                valid_candidates = []

        if not valid_candidates:
            return []

        # Step 2: Remove previously seen
        seen = set()
        if exclude_seen:
            if self.svd:
                seen.update(self.svd.user_seen_recipes.get(user_id, set()))
            if self.item_cf:
                seen.update(self.item_cf.user_seen_recipes.get(user_id, {}).keys())

        unseen_cands = [r for r in valid_candidates if r not in seen]
        if not unseen_cands:
            unseen_cands = valid_candidates[:offset + top_k]

        # Step 3: Fast vectorized hybrid scoring
        # Candidate subsets for ranking (up to 1,500 candidates for deep pagination)
        cands_to_score = unseen_cands[:min(len(unseen_cands), 1500)]

        scored = []
        for r_id in cands_to_score:
            score = self.predict(user_id, r_id)
            scored.append((r_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[offset : offset + top_k]

    def record_interaction(
        self,
        user_id: int,
        recipe_id: int,
        rating: float,
        feedback_type: str = "rating"
    ) -> Dict[str, Any]:
        """
        Adaptive Learning Feedback Loop:
        Immediately adapts the user's latent preference vector (in SVD)
        and taste vector (in Content-Based) in real-time.
        """
        if user_id not in self.user_session_feedback:
            self.user_session_feedback[user_id] = []

        self.user_session_feedback[user_id].append({
            "recipe_id": recipe_id,
            "rating": rating,
            "type": feedback_type
        })

        # Online adaptation in SVD
        if self.svd:
            self.svd.adapt_user(user_id, recipe_id, rating, step_size=self.adaptive_step_size)

        # Online adaptation in Content-Based
        if self.content:
            if user_id not in self.content.user_seen_recipes:
                self.content.user_seen_recipes[user_id] = {}
            self.content.user_seen_recipes[user_id][recipe_id] = rating
            self.content._update_user_profile(user_id, self.content.user_seen_recipes[user_id])

        # Record in Item-CF cache
        if self.item_cf:
            if user_id not in self.item_cf.user_seen_recipes:
                self.item_cf.user_seen_recipes[user_id] = {}
            self.item_cf.user_seen_recipes[user_id][recipe_id] = rating

        return {
            "status": "adapted",
            "user_id": user_id,
            "recipe_id": recipe_id,
            "rating": rating,
            "total_session_interactions": len(self.user_session_feedback[user_id])
        }
