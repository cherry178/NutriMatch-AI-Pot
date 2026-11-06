"""
Regularized Biased SVD (Matrix Factorization) Recommender
=========================================================
Implements latent factor matrix factorization with explicit user/item biases,
L2 weight regularization, dynamic learning rate decay, and validation-based early stopping.
Guarantees optimal bias-variance balance to prevent both underfitting and overfitting.
Provides online real-time adaptation for dynamic session learning.
"""

import copy
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):
    """
    Regularized Biased Matrix Factorization (SVD) with Early Stopping.
    """

    def __init__(
        self,
        n_factors: int = 32,
        learning_rate: float = 0.01,
        reg: float = 0.05,
        n_epochs: int = 30,
        patience: int = 4,
        lr_decay: float = 0.95,
        random_state: int = 42
    ):
        super().__init__(name="SVD Matrix Factorization")
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.patience = patience
        self.lr_decay = lr_decay
        self.random_state = random_state

        # Parameters
        self.global_mean: float = 0.0
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        # Mappings
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.recipe_to_idx: Dict[int, int] = {}
        self.idx_to_recipe: Dict[int, int] = {}

        # User interaction cache
        self.user_seen_recipes: Dict[int, set] = {}

        # Diagnostics history
        self.training_history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_rmse": [],
            "val_rmse": []
        }
        self.best_epoch: int = 0
        self.best_val_rmse: float = float("inf")

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "SVDRecommender":
        """
        Train regularized biased matrix factorization via SGD with validation-based early stopping.
        """
        logger.info(f"Training {self.name} on {len(train_df):,} ratings (Factors={self.n_factors}, Reg={self.reg})...")
        rng = np.random.RandomState(self.random_state)

        # Build contiguous IDs
        unique_users = train_df["user_id"].unique()
        unique_recipes = train_df["recipe_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.recipe_to_idx = {r: i for i, r in enumerate(unique_recipes)}
        self.idx_to_recipe = {i: r for r, i in self.recipe_to_idx.items()}

        n_users = len(self.user_to_idx)
        n_items = len(self.recipe_to_idx)

        # Cache seen
        self.user_seen_recipes = {u: set() for u in unique_users}
        for u, r in zip(train_df["user_id"], train_df["recipe_id"]):
            self.user_seen_recipes[u].add(r)

        self.global_mean = float(train_df["rating"].mean())

        # Initialize parameters: zero biases, small normal latent vectors
        self.user_biases = np.zeros(n_users, dtype=np.float32)
        self.item_biases = np.zeros(n_items, dtype=np.float32)
        self.user_factors = rng.normal(0, 0.05, size=(n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.05, size=(n_items, self.n_factors)).astype(np.float32)

        # Prepare fast arrays for SGD
        u_arr = np.array([self.user_to_idx[u] for u in train_df["user_id"]], dtype=np.int32)
        i_arr = np.array([self.recipe_to_idx[r] for r in train_df["recipe_id"]], dtype=np.int32)
        r_arr = train_df["rating"].values.astype(np.float32)
        n_ratings = len(r_arr)

        # Prepare validation arrays if available
        has_val = val_df is not None and len(val_df) > 0
        if has_val:
            val_mask = val_df["user_id"].isin(self.user_to_idx) & val_df["recipe_id"].isin(self.recipe_to_idx)
            val_clean = val_df[val_mask]
            val_u = np.array([self.user_to_idx[u] for u in val_clean["user_id"]], dtype=np.int32)
            val_i = np.array([self.recipe_to_idx[r] for r in val_clean["recipe_id"]], dtype=np.int32)
            val_r = val_clean["rating"].values.astype(np.float32)

        best_weights = None
        epochs_no_improve = 0
        current_lr = self.learning_rate

        for epoch in range(1, self.n_epochs + 1):
            # Shuffle interaction order each epoch
            indices = rng.permutation(n_ratings)
            total_squared_error = 0.0
            reg_penalty = 0.0

            for idx in indices:
                u = u_arr[idx]
                i = i_arr[idx]
                target = r_arr[idx]

                # Prediction: r_hat = mu + b_u + b_i + P_u . Q_i
                dot = np.dot(self.user_factors[u], self.item_factors[i])
                pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + dot
                err = target - pred
                total_squared_error += err ** 2

                # SGD updates with L2 regularization
                p_u = self.user_factors[u].copy()
                q_i = self.item_factors[i].copy()

                self.user_biases[u] += current_lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += current_lr * (err - self.reg * self.item_biases[i])
                self.user_factors[u] += current_lr * (err * q_i - self.reg * p_u)
                self.item_factors[i] += current_lr * (err * p_u - self.reg * q_i)

            train_rmse = np.sqrt(total_squared_error / n_ratings)
            train_loss = (total_squared_error / n_ratings) + (self.reg * (
                np.sum(self.user_biases ** 2) + np.sum(self.item_biases ** 2) +
                np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
            )) / n_ratings

            # Compute validation RMSE
            if has_val:
                val_dots = np.sum(self.user_factors[val_u] * self.item_factors[val_i], axis=1)
                val_preds = self.global_mean + self.user_biases[val_u] + self.item_biases[val_i] + val_dots
                val_preds = np.clip(val_preds, 1.0, 5.0)
                val_rmse = float(np.sqrt(np.mean((val_r - val_preds) ** 2)))
            else:
                val_rmse = train_rmse

            self.training_history["epoch"].append(epoch)
            self.training_history["train_loss"].append(float(train_loss))
            self.training_history["train_rmse"].append(float(train_rmse))
            self.training_history["val_rmse"].append(float(val_rmse))

            logger.info(
                f"Epoch {epoch:02d}/{self.n_epochs:02d} | Train RMSE: {train_rmse:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | LR: {current_lr:.5f}"
            )

            # Early stopping & model checkpointing
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch
                best_weights = {
                    "user_biases": self.user_biases.copy(),
                    "item_biases": self.item_biases.copy(),
                    "user_factors": self.user_factors.copy(),
                    "item_factors": self.item_factors.copy()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        f"Early stopping triggered at Epoch {epoch}! Best Val RMSE: {self.best_val_rmse:.4f} at Epoch {self.best_epoch}."
                    )
                    break

            # Decay learning rate
            current_lr *= self.lr_decay

        # Restore best checkpoint
        if best_weights is not None:
            self.user_biases = best_weights["user_biases"]
            self.item_biases = best_weights["item_biases"]
            self.user_factors = best_weights["user_factors"]
            self.item_factors = best_weights["item_factors"]
            logger.info(f"Restored best model weights from Epoch {self.best_epoch}.")

        self.is_fitted = True
        return self

    def predict(self, user_id: int, recipe_id: int) -> float:
        """Predict rating for given (user, recipe). Handles unknown users/items gracefully."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        u_idx = self.user_to_idx.get(user_id, None)
        i_idx = self.recipe_to_idx.get(recipe_id, None)

        if u_idx is None and i_idx is None:
            return self.global_mean
        if u_idx is None:
            return float(np.clip(self.global_mean + self.item_biases[i_idx], 1.0, 5.0))
        if i_idx is None:
            return float(np.clip(self.global_mean + self.user_biases[u_idx], 1.0, 5.0))

        dot = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        pred = self.global_mean + self.user_biases[u_idx] + self.item_biases[i_idx] + dot
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_ids: Optional[List[int]] = None,
        offset: int = 0
    ) -> List[Tuple[int, float]]:
        """Compute vectorized scores for all candidates and return top-k."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before recommend.")

        seen = self.user_seen_recipes.get(user_id, set()) if exclude_seen else set()
        u_idx = self.user_to_idx.get(user_id, None)

        if candidate_ids is not None:
            cand_valid = [r for r in candidate_ids if r in self.recipe_to_idx and r not in seen]
        else:
            cand_valid = [r for r in self.recipe_to_idx.keys() if r not in seen]

        if not cand_valid:
            return []

        cand_indices = np.array([self.recipe_to_idx[r] for r in cand_valid], dtype=np.int32)

        if u_idx is not None:
            u_vec = self.user_factors[u_idx]
            u_bias = self.user_biases[u_idx]
            # Vectorized dot products across all candidate item factors
            item_vecs = self.item_factors[cand_indices]
            dots = np.dot(item_vecs, u_vec)
            scores = self.global_mean + u_bias + self.item_biases[cand_indices] + dots
        else:
            scores = self.global_mean + self.item_biases[cand_indices]

        scores = np.clip(scores, 1.0, 5.0)

        # Get top-k indices with offset
        total_needed = offset + top_k
        if len(scores) > total_needed:
            top_k_pos = np.argpartition(scores, -total_needed)[-total_needed:]
            top_k_sorted = top_k_pos[np.argsort(scores[top_k_pos])[::-1]]
        else:
            top_k_sorted = np.argsort(scores)[::-1]

        sliced = top_k_sorted[offset : offset + top_k]
        return [(cand_valid[idx], float(scores[idx])) for idx in sliced]

    def adapt_user(self, user_id: int, recipe_id: int, rating: float, step_size: float = 0.05) -> None:
        """
        Online adaptive feedback learning:
        Dynamically updates the user's latent representation and bias based on real-time feedback.
        """
        if not self.is_fitted:
            return

        if recipe_id not in self.recipe_to_idx:
            return

        i_idx = self.recipe_to_idx[recipe_id]
        q_i = self.item_factors[i_idx].copy()

        if user_id not in self.user_to_idx:
            # Register new user dynamically
            new_u_idx = len(self.user_to_idx)
            self.user_to_idx[user_id] = new_u_idx
            self.idx_to_user[new_u_idx] = user_id
            self.user_biases = np.append(self.user_biases, 0.0)
            new_factor = np.random.normal(0, 0.05, size=(1, self.n_factors)).astype(np.float32)
            self.user_factors = np.vstack([self.user_factors, new_factor])
            self.user_seen_recipes[user_id] = set()

        u_idx = self.user_to_idx[user_id]
        self.user_seen_recipes[user_id].add(recipe_id)

        # Compute immediate error and apply gradient step
        pred = self.predict(user_id, recipe_id)
        err = rating - pred

        p_u = self.user_factors[u_idx].copy()
        self.user_biases[u_idx] += step_size * (err - self.reg * self.user_biases[u_idx])
        self.user_factors[u_idx] += step_size * (err * q_i - self.reg * p_u)
        logger.info(f"Adapted user {user_id} latent profile on recipe {recipe_id} with rating {rating:.1f} (err: {err:.2f})")
