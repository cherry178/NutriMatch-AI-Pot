"""
Item-Based Collaborative Filtering Recommender
==============================================
Implements neighborhood-based Item-Item Collaborative Filtering with shrinkage regularization.
Finds recipes similar to those a user previously rated/liked and aggregates neighbor preferences.
Shrinkage penalty suppresses spurious high similarities on item pairs with few co-ratings,
preventing overfitting on sparse tails.
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class ItemBasedCFRecommender(BaseRecommender):
    """
    Item-Based Collaborative Filtering Recommender with Shrinkage Regularization.
    """

    def __init__(
        self,
        k_neighbors: int = 30,
        shrinkage: float = 10.0,
        min_common_users: int = 2
    ):
        super().__init__(name="Item-Based CF")
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.min_common_users = min_common_users

        # Mappings
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.recipe_to_idx: Dict[int, int] = {}
        self.idx_to_recipe: Dict[int, int] = {}

        # Similarity matrix and caches
        self.item_similarity_csr: Optional[csr_matrix] = None
        self.user_seen_recipes: Dict[int, Dict[int, float]] = {}
        self.user_means: Dict[int, float] = {}
        self.global_mean: float = 3.5

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "ItemBasedCFRecommender":
        """Fit item-item similarity matrix with shrinkage."""
        logger.info(f"Fitting {self.name} on {len(train_df):,} interactions...")
        self.global_mean = float(train_df["rating"].mean())

        # Construct contiguous IDs
        unique_users = train_df["user_id"].unique()
        unique_recipes = train_df["recipe_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.recipe_to_idx = {r: i for i, r in enumerate(unique_recipes)}
        self.idx_to_recipe = {i: r for r, i in self.recipe_to_idx.items()}

        n_users = len(self.user_to_idx)
        n_items = len(self.recipe_to_idx)

        # Cache seen ratings per user
        self.user_seen_recipes = {}
        for _, row in train_df.iterrows():
            u, r, rat = int(row["user_id"]), int(row["recipe_id"]), float(row["rating"])
            if u not in self.user_seen_recipes:
                self.user_seen_recipes[u] = {}
            self.user_seen_recipes[u][r] = rat

        # User mean ratings
        self.user_means = {
            u: float(np.mean(list(rats.values()))) for u, rats in self.user_seen_recipes.items()
        }

        # Build Sparse User-Item Matrix (n_items x n_users for fast item-item cosine)
        rows = [self.recipe_to_idx[r] for r in train_df["recipe_id"]]
        cols = [self.user_to_idx[u] for u in train_df["user_id"]]
        vals = train_df["rating"].values.astype(np.float32)

        # Item-user sparse matrix
        item_user_mat = csr_matrix((vals, (rows, cols)), shape=(n_items, n_users), dtype=np.float32)

        # Binary co-rating count matrix: item_user_bin @ item_user_bin.T
        item_user_bin = csr_matrix((np.ones_like(vals), (rows, cols)), shape=(n_items, n_users), dtype=np.float32)
        co_counts = item_user_bin.dot(item_user_bin.T).toarray()

        logger.info("Computing item-item cosine similarity...")
        raw_sim = cosine_similarity(item_user_mat, dense_output=False)

        # Apply shrinkage regularization: s_shrunk = (n_co / (n_co + alpha)) * raw_sim
        raw_sim_arr = raw_sim.toarray()
        shrinkage_factor = co_counts / (co_counts + self.shrinkage + 1e-8)
        shrunk_sim = raw_sim_arr * shrinkage_factor

        # Set diagonal to zero (an item is not its own neighbor for recommendation)
        np.fill_diagonal(shrunk_sim, 0.0)

        # Retain top-k neighbors per item to maintain sparsity and prevent noisy neighbors
        logger.info(f"Filtering top-{self.k_neighbors} neighbors per item...")
        for i in range(n_items):
            row_sims = shrunk_sim[i]
            if len(row_sims) > self.k_neighbors:
                top_k_indices = np.argpartition(row_sims, -self.k_neighbors)[-self.k_neighbors:]
                mask = np.ones(n_items, dtype=bool)
                mask[top_k_indices] = False
                shrunk_sim[i, mask] = 0.0

        self.item_similarity_csr = csr_matrix(shrunk_sim, dtype=np.float32)
        self.is_fitted = True
        logger.info(f"{self.name} fitted successfully with {n_items:,} items and {n_users:,} users.")
        return self

    def predict(self, user_id: int, recipe_id: int) -> float:
        """Predict rating for (user, recipe)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        if user_id not in self.user_seen_recipes or recipe_id not in self.recipe_to_idx:
            user_mean = self.user_means.get(user_id, self.global_mean)
            return user_mean

        rec_idx = self.recipe_to_idx[recipe_id]
        user_ratings = self.user_seen_recipes[user_id]

        # Neighbor weights
        sim_row = self.item_similarity_csr[rec_idx].toarray().flatten()

        sim_sum = 0.0
        weighted_sum = 0.0

        for seen_rec_id, rating in user_ratings.items():
            if seen_rec_id in self.recipe_to_idx:
                seen_idx = self.recipe_to_idx[seen_rec_id]
                sim = sim_row[seen_idx]
                if sim > 0:
                    sim_sum += sim
                    weighted_sum += sim * rating

        if sim_sum > 1e-6:
            pred = weighted_sum / sim_sum
        else:
            pred = self.user_means.get(user_id, self.global_mean)

        return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_ids: Optional[List[int]] = None,
        offset: int = 0
    ) -> List[Tuple[int, float]]:
        """Generate top-k recommendations for user."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before recommend.")

        seen = self.user_seen_recipes.get(user_id, {})
        seen_set = set(seen.keys()) if exclude_seen else set()

        if candidate_ids is not None:
            candidates = [r for r in candidate_ids if r in self.recipe_to_idx and r not in seen_set]
        else:
            candidates = [r for r in self.recipe_to_idx.keys() if r not in seen_set]

        if not candidates:
            return []

        scores = [(r, self.predict(user_id, r)) for r in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[offset : offset + top_k]

    def get_similar_recipes(self, recipe_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return items most similar to a given recipe (Item-to-Item)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before querying similar recipes.")
        if recipe_id not in self.recipe_to_idx:
            return []

        rec_idx = self.recipe_to_idx[recipe_id]
        sim_row = self.item_similarity_csr[rec_idx].toarray().flatten()

        top_indices = np.argsort(sim_row)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sim_row[idx])
            if score > 0:
                results.append((self.idx_to_recipe[idx], score))
        return results
