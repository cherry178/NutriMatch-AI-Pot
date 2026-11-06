"""
End-to-End Pipeline Runner
==========================
Trains all recommender models (Item-Based CF, Regularized SVD, Content-Based, Adaptive Hybrid),
executes comprehensive evaluation on held-out test data, saves model checkpoints, and exports
benchmark reports and overfitting/underfitting diagnostic curves.
"""

import os
import sys
import json
import logging
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.utils.logger import setup_logger
from src.models.item_cf import ItemBasedCFRecommender
from src.models.svd_recommender import SVDRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.knowledge_filter import KnowledgeConstraintFilter
from src.models.hybrid_recommender import AdaptiveHybridRecommender
from src.evaluation.evaluator import BenchmarkEvaluator

logger = setup_logger("run_pipeline")


def run_pipeline():
    logger.info("=" * 70)
    logger.info("STARTING ADAPTIVE PERSONALIZED RECIPE RECOMMENDATION ENGINE PIPELINE")
    logger.info("=" * 70)

    # 1. Load Data
    train_path = "data/processed/train.parquet"
    val_path = "data/processed/val.parquet"
    test_path = "data/processed/test.parquet"
    recipes_path = "data/processed/recipes_clean.parquet"

    assert os.path.exists(train_path), f"Train dataset not found: {train_path}"
    assert os.path.exists(test_path), f"Test dataset not found: {test_path}"
    assert os.path.exists(recipes_path), f"Recipes dataset not found: {recipes_path}"

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    df_recipes = pd.read_parquet(recipes_path)

    logger.info(f"Loaded Train: {len(df_train):,} rows | Val: {len(df_val):,} rows | Test: {len(df_test):,} rows")
    logger.info(f"Loaded Recipes: {len(df_recipes):,} recipes across {df_recipes['RecipeCategory'].nunique()} categories")

    os.makedirs("models_saved", exist_ok=True)

    # 2. Train Item-Based CF
    logger.info("-" * 50)
    logger.info("Training Algorithm 1: Item-Based Collaborative Filtering (with Shrinkage)...")
    t0 = time.time()
    item_cf = ItemBasedCFRecommender(k_neighbors=30, shrinkage=10.0)
    item_cf.fit(df_train)
    logger.info(f"Item-Based CF trained in {time.time() - t0:.2f} seconds.")
    item_cf.save("models_saved/item_cf.joblib")

    # 3. Train Regularized SVD (Matrix Factorization)
    logger.info("-" * 50)
    logger.info("Training Algorithm 2: Singular Value Decomposition (SVD Matrix Factorization)...")
    t0 = time.time()
    svd = SVDRecommender(
        n_factors=32,
        learning_rate=0.015,
        reg=0.05,
        n_epochs=25,
        patience=4,
        lr_decay=0.96,
        random_state=42
    )
    svd.fit(df_train, val_df=df_val)
    logger.info(f"SVD Matrix Factorization trained in {time.time() - t0:.2f} seconds.")
    svd.save("models_saved/svd_model.joblib")

    # Save SVD Diagnostics (Loss, Train RMSE, Val RMSE)
    diagnostics = {
        "model": "SVD Matrix Factorization",
        "best_epoch": svd.best_epoch,
        "best_val_rmse": round(svd.best_val_rmse, 4),
        "history": svd.training_history
    }
    with open("models_saved/diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Saved SVD training diagnostics to models_saved/diagnostics.json.")

    # 4. Train Content-Based Recommender
    logger.info("-" * 50)
    logger.info("Training Algorithm 3: Content-Based Recommender (TF-IDF)...")
    t0 = time.time()
    content_model = ContentBasedRecommender(max_features=4000)
    content_model.fit(df_train, df_recipes)
    logger.info(f"Content-Based Recommender trained in {time.time() - t0:.2f} seconds.")
    content_model.save("models_saved/content_model.joblib")

    # 5. Build Knowledge Filter & Adaptive Hybrid Recommender
    logger.info("-" * 50)
    logger.info("Building Algorithm 4: Adaptive Hybrid Recommendation Engine...")
    knowledge_filter = KnowledgeConstraintFilter(df_recipes)
    hybrid = AdaptiveHybridRecommender(
        svd_model=svd,
        item_cf_model=item_cf,
        content_model=content_model,
        knowledge_filter=knowledge_filter,
        svd_weight=0.50,
        item_cf_weight=0.30,
        content_weight=0.20,
        adaptive_step_size=0.05
    )
    hybrid.save("models_saved/hybrid_model.joblib")
    logger.info("Adaptive Hybrid Recommender configured and saved.")

    # 6. Comparative Model Evaluation
    logger.info("-" * 50)
    logger.info("Running Comparative Benchmark on Held-out Test Set...")
    evaluator = BenchmarkEvaluator(
        test_df=df_test,
        positive_rating_threshold=4.0,
        top_k_list=[5, 10, 20],
        sample_users=400,
        random_seed=42
    )

    models_to_benchmark = [item_cf, svd, content_model, hybrid]
    benchmark_report = evaluator.compare_models(
        models=models_to_benchmark,
        total_catalog_recipes=len(df_recipes),
        output_file="models_saved/benchmark_results.json"
    )

    logger.info("=" * 70)
    logger.info(f"BENCHMARK COMPLETE! Champion Algorithm: {benchmark_report['champion_model']}")
    logger.info("=" * 70)
    for model_name, metrics in benchmark_report["metrics"].items():
        logger.info(
            f"[{model_name:28s}] NDCG@10: {metrics.get('ndcg@10'):.4f} | "
            f"Recall@10: {metrics.get('recall@10'):.4f} | "
            f"Precision@10: {metrics.get('precision@10'):.4f} | "
            f"RMSE: {metrics.get('rmse'):.4f}"
        )


if __name__ == "__main__":
    run_pipeline()
