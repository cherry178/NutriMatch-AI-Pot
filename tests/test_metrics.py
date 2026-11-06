"""
Unit Tests for Recommendation Metrics
=====================================
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
    hit_rate_at_k,
    compute_rmse_mae,
    compute_catalog_coverage
)


def test_precision_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    gt = {2, 4, 6}

    # k=2: recs=[1,2], hit=1 -> P@2=0.5, R@2=1/3
    assert precision_at_k(recs, gt, 2) == 0.5
    assert recall_at_k(recs, gt, 2) == pytest.approx(1 / 3)

    # k=5: recs=[1,2,3,4,5], hits=2 -> P@5=0.4, R@5=2/3
    assert precision_at_k(recs, gt, 5) == 0.4
    assert recall_at_k(recs, gt, 5) == pytest.approx(2 / 3)


def test_ndcg_at_k():
    # Perfect ranking
    recs = [10, 20, 30]
    gt = {10, 20}
    assert ndcg_at_k(recs, gt, 2) == 1.0

    # Reverse ranking
    recs_rev = [99, 10, 20]
    ndcg_val = ndcg_at_k(recs_rev, gt, 3)
    assert 0.0 < ndcg_val < 1.0


def test_hit_rate_at_k():
    assert hit_rate_at_k([1, 2, 3], {2, 5}, 3) == 1.0
    assert hit_rate_at_k([1, 4, 3], {2, 5}, 3) == 0.0


def test_rmse_mae():
    actuals = [5.0, 4.0, 3.0]
    preds = [4.0, 4.0, 4.0]
    res = compute_rmse_mae(actuals, preds)
    # errors: 1.0, 0.0, -1.0 -> squared errors: 1, 0, 1 -> mean=2/3 -> sqrt=0.816
    assert res["rmse"] == pytest.approx(np.sqrt(2 / 3), rel=1e-3)
    assert res["mae"] == pytest.approx(2 / 3, rel=1e-3)


def test_catalog_coverage():
    all_recs = [[1, 2, 3], [3, 4, 5]]
    coverage = compute_catalog_coverage(all_recs, total_catalog_items=10)
    # unique: {1, 2, 3, 4, 5} -> 5 / 10 = 0.5
    assert coverage == 0.5
