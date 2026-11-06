"""
Base Recommender Interface
==========================
Defines the standard interface and common persistence methods for all recommendation models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
import os
import joblib
import pandas as pd
import numpy as np


class BaseRecommender(ABC):
    """Abstract base class for all recommender systems."""

    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "BaseRecommender":
        """Train the recommendation model on training interactions."""
        pass

    @abstractmethod
    def predict(self, user_id: int, recipe_id: int) -> float:
        """Predict the preference score / rating for a single (user, recipe) pair."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_ids: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Return top-k recipe recommendations for a user as a list of (recipe_id, score) tuples.
        """
        pass

    def save(self, filepath: str) -> None:
        """Persist model artifact using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaseRecommender":
        """Load persisted model artifact."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
