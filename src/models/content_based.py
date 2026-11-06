"""
Content-Based Recipe Recommender
================================
Uses TF-IDF vectorization across recipe ingredients, tags, categories, and descriptions.
Constructs user preference taste profiles by aggregating rating-weighted feature vectors of consumed recipes.
Recommends candidate recipes with high cosine similarity to the user's taste vector.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """
    TF-IDF Vector Space Content-Based Recommender for culinary recipes.
    """

    def __init__(
        self,
        max_features: int = 4000,
        stop_words: str = "english",
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        super().__init__(name="Content-Based (TF-IDF)")
        self.max_features = max_features
        self.stop_words = stop_words
        self.ngram_range = ngram_range

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            sublinear_tf=True
        )
        self.tfidf_matrix: Optional[csr_matrix] = None
        self.recipe_to_idx: Dict[int, int] = {}
        self.idx_to_recipe: Dict[int, int] = {}
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.user_seen_recipes: Dict[int, Dict[int, float]] = {}

    def fit(self, train_df: pd.DataFrame, recipes_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Fit TF-IDF matrix on recipes and build user taste vectors from training interactions."""
        logger.info(f"Fitting {self.name} on {len(recipes_df):,} recipe text profiles...")

        # Map recipes
        recipes_clean = recipes_df.drop_duplicates(subset=["RecipeId"]).copy()
        self.recipe_to_idx = {int(r): i for i, r in enumerate(recipes_clean["RecipeId"])}
        self.idx_to_recipe = {i: r for r, i in self.recipe_to_idx.items()}

        corpus = recipes_clean["content_text"].fillna("").tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info(f"TF-IDF matrix vocabulary size: {self.tfidf_matrix.shape[1]:,} features.")

        # Cache seen ratings
        self.user_seen_recipes = {}
        for _, row in train_df.iterrows():
            u, r, rat = int(row["user_id"]), int(row["recipe_id"]), float(row["rating"])
            if u not in self.user_seen_recipes:
                self.user_seen_recipes[u] = {}
            self.user_seen_recipes[u][r] = rat

        # Build user profile vectors: weighted centroid of rated recipes
        logger.info(f"Building user profile vectors for {len(self.user_seen_recipes):,} users...")
        for u, rec_ratings in self.user_seen_recipes.items():
            self._update_user_profile(u, rec_ratings)

        self.is_fitted = True
        return self

    def _update_user_profile(self, user_id: int, rec_ratings: Dict[int, float]) -> None:
        """Construct user profile vector as rating-weighted mean of recipe TF-IDF vectors."""
        vecs = []
        weights = []
        for r_id, rating in rec_ratings.items():
            if r_id in self.recipe_to_idx:
                idx = self.recipe_to_idx[r_id]
                vec = self.tfidf_matrix[idx].toarray().flatten()
                # Center weights around 3.0 so disliked recipes (<3) push away
                w = rating - 2.5
                vecs.append(vec)
                weights.append(w)

        if vecs and np.sum(np.abs(weights)) > 0:
            weighted_profile = np.average(vecs, axis=0, weights=weights)
            norm = np.linalg.norm(weighted_profile)
            if norm > 1e-6:
                weighted_profile /= norm
            self.user_profiles[user_id] = weighted_profile.astype(np.float32)

    def predict(self, user_id: int, recipe_id: int) -> float:
        """Predict cosine similarity scaled to rating scale [1.0, 5.0]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        if user_id not in self.user_profiles or recipe_id not in self.recipe_to_idx:
            return 3.5

        u_vec = self.user_profiles[user_id]
        rec_idx = self.recipe_to_idx[recipe_id]
        r_vec = self.tfidf_matrix[rec_idx].toarray().flatten()

        sim = float(np.dot(u_vec, r_vec))
        # Map cosine [-1, 1] or [0, 1] to 1.0 - 5.0
        scaled = 1.0 + 4.0 * max(0.0, sim)
        return float(np.clip(scaled, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_ids: Optional[List[int]] = None,
        offset: int = 0
    ) -> List[Tuple[int, float]]:
        """Recommend recipes with highest cosine similarity to user taste profile."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before recommend.")

        if user_id not in self.user_profiles:
            return []

        u_vec = self.user_profiles[user_id].reshape(1, -1)
        seen = set(self.user_seen_recipes.get(user_id, {}).keys()) if exclude_seen else set()

        if candidate_ids is not None:
            cand_valid = [r for r in candidate_ids if r in self.recipe_to_idx and r not in seen]
        else:
            cand_valid = [r for r in self.recipe_to_idx.keys() if r not in seen]

        if not cand_valid:
            return []

        cand_indices = [self.recipe_to_idx[r] for r in cand_valid]
        cand_matrix = self.tfidf_matrix[cand_indices]

        sims = cosine_similarity(u_vec, cand_matrix).flatten()
        # Scale to 1.0 - 5.0
        scores = 1.0 + 4.0 * np.clip(sims, 0.0, 1.0)

        total_needed = offset + top_k
        if len(scores) > total_needed:
            top_pos = np.argpartition(scores, -total_needed)[-total_needed:]
            top_sorted = top_pos[np.argsort(scores[top_pos])[::-1]]
        else:
            top_sorted = np.argsort(scores)[::-1]

        sliced = top_sorted[offset : offset + top_k]
        return [(cand_valid[i], float(scores[i])) for i in sliced]

    def get_content_similar(self, recipe_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return recipes most similar in ingredients and tags to target recipe."""
        if not self.is_fitted or recipe_id not in self.recipe_to_idx:
            return []

        idx = self.recipe_to_idx[recipe_id]
        target_vec = self.tfidf_matrix[idx]
        sims = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        sims[idx] = -1.0  # Exclude self
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.idx_to_recipe[i], float(sims[i])) for i in top_indices if sims[i] > 0]
