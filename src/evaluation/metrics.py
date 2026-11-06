"""
Recommendation Evaluation Metrics
=================================
Implements industry-standard ranking and rating accuracy metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- HitRate@K
- RMSE & MAE
- Catalog Coverage
"""

import numpy as np
from typing import List, Set, Dict, Any


def precision_at_k(recommended_ids: List[int], ground_truth_set: Set[int], k: int) -> float:
    """Precision@K = (# of relevant items in top-k) / k."""
    if k <= 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = sum(1 for item in rec_k if item in ground_truth_set)
    return float(hits / k)


def recall_at_k(recommended_ids: List[int], ground_truth_set: Set[int], k: int) -> float:
    """Recall@K = (# of relevant items in top-k) / (total # of relevant items)."""
    if not ground_truth_set:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = sum(1 for item in rec_k if item in ground_truth_set)
    return float(hits / len(ground_truth_set))


def ndcg_at_k(recommended_ids: List[int], ground_truth_set: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    if not ground_truth_set or k <= 0:
        return 0.0

    rec_k = recommended_ids[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in ground_truth_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # Ideal DCG
    n_rel = min(k, len(ground_truth_set))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def average_precision_at_k(recommended_ids: List[int], ground_truth_set: Set[int], k: int) -> float:
    """Average Precision at K for a single user."""
    if not ground_truth_set or k <= 0:
        return 0.0

    rec_k = recommended_ids[:k]
    hits = 0
    sum_precisions = 0.0

    for i, item in enumerate(rec_k):
        if item in ground_truth_set:
            hits += 1
            sum_precisions += hits / (i + 1)

    n_rel = min(k, len(ground_truth_set))
    return float(sum_precisions / n_rel) if n_rel > 0 else 0.0


def hit_rate_at_k(recommended_ids: List[int], ground_truth_set: Set[int], k: int) -> float:
    """HitRate@K: 1.0 if at least one relevant item is in top-K, else 0.0."""
    rec_k = recommended_ids[:k]
    return 1.0 if any(item in ground_truth_set for item in rec_k) else 0.0


def compute_rmse_mae(actuals: List[float], predictions: List[float]) -> Dict[str, float]:
    """Compute Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    if not actuals or not predictions or len(actuals) != len(predictions):
        return {"rmse": 0.0, "mae": 0.0}

    act = np.array(actuals, dtype=np.float64)
    pred = np.array(predictions, dtype=np.float64)

    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    mae = float(np.mean(np.abs(act - pred)))
    return {"rmse": rmse, "mae": mae}


def compute_catalog_coverage(all_recommendations: List[List[int]], total_catalog_items: int) -> float:
    """Fraction of unique items recommended across all users."""
    if total_catalog_items <= 0:
        return 0.0
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    return float(len(unique_recommended) / total_catalog_items)
