"""
Comparative Model Evaluator
===========================
Evaluates and benchmarks recommendation algorithms (Item-Based CF, SVD, Content-Based, Adaptive Hybrid)
on held-out test interactions. Computes ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K)
and rating error metrics (RMSE, MAE). Identifies the champion model and exports benchmark logs.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.models.base import BaseRecommender
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    compute_rmse_mae,
    compute_catalog_coverage
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Evaluates recommender models on held-out test sets.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        positive_rating_threshold: float = 4.0,
        top_k_list: List[int] = [5, 10, 20],
        sample_users: Optional[int] = 500,
        random_seed: int = 42
    ):
        self.test_df = test_df.copy()
        self.positive_threshold = positive_rating_threshold
        self.top_k_list = top_k_list
        self.sample_users = sample_users
        self.random_seed = random_seed

        # Pre-group test ground truth per user
        self.user_test_ratings: Dict[int, Dict[int, float]] = {}
        self.user_positive_items: Dict[int, set] = {}

        for _, row in self.test_df.iterrows():
            u, r, rat = int(row["user_id"]), int(row["recipe_id"]), float(row["rating"])
            if u not in self.user_test_ratings:
                self.user_test_ratings[u] = {}
                self.user_positive_items[u] = set()

            self.user_test_ratings[u][r] = rat
            if rat >= self.positive_threshold:
                self.user_positive_items[u].add(r)

        # Candidate user pool: users who have at least one positive test item
        eval_candidates = [u for u, items in self.user_positive_items.items() if len(items) > 0]
        if self.sample_users and len(eval_candidates) > self.sample_users:
            rng = np.random.RandomState(self.random_seed)
            self.eval_users = list(rng.choice(eval_candidates, size=self.sample_users, replace=False))
        else:
            self.eval_users = eval_candidates

        logger.info(f"BenchmarkEvaluator initialized with {len(self.eval_users):,} test evaluation users.")

    def evaluate_model(self, model: BaseRecommender, total_catalog_recipes: int = 4179) -> Dict[str, Any]:
        """Evaluate a single recommender model across all metrics."""
        logger.info(f"Evaluating model: {model.name}...")

        precisions = {k: [] for k in self.top_k_list}
        recalls = {k: [] for k in self.top_k_list}
        ndcgs = {k: [] for k in self.top_k_list}
        map_10_list = []
        hit_10_list = []
        all_recs = []

        actual_ratings = []
        pred_ratings = []

        for u in self.eval_users:
            gt_pos = self.user_positive_items[u]

            # Rating accuracy predictions
            for r_id, actual in self.user_test_ratings[u].items():
                try:
                    pred = model.predict(u, r_id)
                    actual_ratings.append(actual)
                    pred_ratings.append(pred)
                except Exception:
                    pass

            # Top-K recommendation ranking
            max_k = max(self.top_k_list)
            try:
                rec_tuples = model.recommend(u, top_k=max_k, exclude_seen=True)
                rec_ids = [r for r, _ in rec_tuples]
            except Exception as e:
                rec_ids = []

            all_recs.append(rec_ids)

            for k in self.top_k_list:
                precisions[k].append(precision_at_k(rec_ids, gt_pos, k))
                recalls[k].append(recall_at_k(rec_ids, gt_pos, k))
                ndcgs[k].append(ndcg_at_k(rec_ids, gt_pos, k))

            map_10_list.append(average_precision_at_k(rec_ids, gt_pos, 10))
            hit_10_list.append(hit_rate_at_k(rec_ids, gt_pos, 10))

        # Rating metrics
        rating_metrics = compute_rmse_mae(actual_ratings, pred_ratings)
        coverage = compute_catalog_coverage(all_recs, total_catalog_recipes)

        results = {
            "model_name": model.name,
            "rmse": round(rating_metrics["rmse"], 4),
            "mae": round(rating_metrics["mae"], 4),
            "map@10": round(float(np.mean(map_10_list)), 4),
            "hit_rate@10": round(float(np.mean(hit_10_list)), 4),
            "catalog_coverage": round(coverage, 4),
        }

        for k in self.top_k_list:
            results[f"precision@{k}"] = round(float(np.mean(precisions[k])), 4)
            results[f"recall@{k}"] = round(float(np.mean(recalls[k])), 4)
            results[f"ndcg@{k}"] = round(float(np.mean(ndcgs[k])), 4)

        logger.info(f"{model.name} Results -> NDCG@10: {results.get('ndcg@10'):.4f} | Recall@10: {results.get('recall@10'):.4f} | RMSE: {results.get('rmse'):.4f}")
        return results

    def compare_models(
        self,
        models: List[BaseRecommender],
        total_catalog_recipes: int = 4179,
        output_file: Optional[str] = "models_saved/benchmark_results.json"
    ) -> Dict[str, Any]:
        """Benchmark all models, rank them, and determine champion algorithm."""
        all_results = {}
        for m in models:
            res = self.evaluate_model(m, total_catalog_recipes=total_catalog_recipes)
            all_results[m.name] = res

        # Determine champion based on composite score: NDCG@10 (60%) + Recall@10 (40%)
        ranked = sorted(
            all_results.items(),
            key=lambda x: (x[1].get("ndcg@10", 0) * 0.6 + x[1].get("recall@10", 0) * 0.4),
            reverse=True
        )
        champion_name = ranked[0][0]

        summary = {
            "champion_model": champion_name,
            "ranking": [name for name, _ in ranked],
            "metrics": all_results
        }

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved benchmark results to {output_file}")

        return summary
