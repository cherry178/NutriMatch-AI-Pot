# Adaptive Personalized Recipe Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![RecBole](https://img.shields.io/badge/RecBole-Dataset_Standard-orange.svg)](https://recbole.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An industry-grade, production-ready recipe recommendation engine based on the **Food.com Recipes and User Interactions dataset** referenced in [RUCAIBox/RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) and RecBole.

---

## 👥 Project Team & Authors
* **Krrish Jha** (24STUCHH010116)
* **Rasineni Charishma** (24STUCHH010662)
* **Ande Hiranmai** (24STUCHH010931)
* **Chirra Chaithanya** (24STUCHH010120)

---

## 📖 Abstract & Problem Statement
Today, people use numerous online recipe platforms. With an enormous catalog of available recipes, users struggle to discover culinary choices that align with their personal taste preferences, nutritional targets, and dietary constraints. Traditional recommendation systems routinely over-recommend globally popular dishes while failing to account for individual palate histories.

The **Adaptive Personalized Recipe Recommendation Engine** solves this challenge through an intelligent multi-layer recommender architecture. Using the benchmark **Food dataset** (sourced from Food.com and the RUCAIBox recommender benchmark repository), the system studies historical user–recipe interaction patterns and deploys two core algorithms:
1. **Item-Based Collaborative Filtering (Item-CF)**: Recommends recipes that are behaviorally similar to those the user has previously cooked and enjoyed, utilizing co-rating shrinkage regularization to avoid tail-sparsity noise.
2. **Singular Value Decomposition (SVD / Matrix Factorization)**: Decomposes the user–recipe rating matrix into latent preference and flavor profile embeddings, with explicit user/item biases, $L_2$ weight regularization, and validation-based early stopping.

Both models, along with **Content-Based Filtering (TF-IDF)** and a multi-stage **Adaptive Hybrid Engine**, are empirically evaluated on a held-out test split using industry-standard ranking metrics (**Precision@K**, **Recall@K**, **NDCG@K**, **MAP@K**, **Hit Rate@K**) and rating error (**RMSE**, **MAE**). The better-performing algorithm is selected for the production engine, which features real-time dynamic adaptation to incoming session feedback.

---

## 🏗️ System Architecture & Multi-Layer Engine

```
                               ┌──────────────────────────────────────────────┐
                               │             USER QUERY / PROFILE             │
                               │ (User ID, Active Session, Dietary Filters)   │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                       ┌──────────────────────────────────────────────────────────────┐
                       │           STAGE 1: KNOWLEDGE-BASED CONSTRAINT FILTER         │
                       │ (Vegetarian, Vegan, Gluten-Free, Low-Calorie, Prep Time < 30m│
                       │           Prunes non-compliant candidate recipes)            │
                       └──────────────────────┬───────────────────────┬───────────────┘
                                              │                       │
                       ┌──────────────────────▼───────┐ ┌─────────────▼───────────────┐
                       │  STAGE 2A: COLLABORATIVE     │ │ STAGE 2B: CONTENT-BASED     │
                       │          FILTERING           │ │          FILTERING          │
                       │ ──────────────────────────── │ │ ─────────────────────────── │
                       │ • Regularized SVD (Latent)   │ │ • TF-IDF on Ingredients &   │
                       │ • Item-Based CF (Neighbors)  │ │   Keywords + Cosine Sim     │
                       └──────────────────────┬───────┘ └─────────────┬───────────────┘
                                              │                       │
                                              └───────────┬───────────┘
                                                          ▼
                       ┌──────────────────────────────────────────────────────────────┐
                       │           STAGE 3: ADAPTIVE HYBRID ENSEMBLE SCORING          │
                       │    Score = w_svd * SVD + w_item * ItemCF + w_content * TFIDF │
                       └──────────────────────────────┬───────────────────────────────┘
                                                      │
                                                      ▼
                       ┌──────────────────────────────────────────────────────────────┐
                       │              STAGE 4: TOP-K RANKED RECOMMENDATIONS           │
                       └──────────────────────────────┬───────────────────────────────┘
                                                      │
                                                      ▼ User interacts / rates (1-5★)
                       ┌──────────────────────────────────────────────────────────────┐
                       │             ONLINE ADAPTIVE FEEDBACK LEARNING LOOP           │
                       │  p_u <- p_u + eta*(r - r_hat)*q_i  (Real-Time Profile Shift) │
                       └──────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Anti-Underfitting & Anti-Overfitting Safeguards

A major requirement for this project is guaranteeing **no underfitting and no overfitting**:

### 1. Preventing Underfitting (High Bias)
* **Sufficient Model Capacity**: Employs $k=32$ latent factors in SVD, enabling the model to capture multi-faceted palate preferences (sweet vs savory, quick-cook vs slow braise, spicy vs mild).
* **Explicit Bias Partitioning**: Ratings are decomposed into $\mu + b_u + b_i + p_u^T q_i$, capturing user baseline generosity and universal recipe appeal before latent interaction modeling.
* **Hybrid Synergy**: Content-based TF-IDF and knowledge rules compensate for cold-start sparsity where collaborative signals are weak.

### 2. Preventing Overfitting (High Variance)
* **$L_2$ Weight Regularization ($\lambda = 0.05$)**: The objective function penalizes large embedding norms:
  $$\mathcal{L}_{reg} = \sum_{(u,i) \in Train} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left( b_u^2 + b_i^2 + \|p_u\|_2^2 + \|q_i\|_2^2 \right)$$
* **Item-CF Shrinkage Penalty ($\alpha = 10.0$)**: Dampens spurious high cosine similarities between recipe pairs that share only 1 or 2 ratings:
  $$s_{ij}^{shrunk} = \frac{n_{ij}}{n_{ij} + \alpha} \cdot \cos(\mathbf{r}_i, \mathbf{r}_j)$$
* **Validation-Based Early Stopping**: Training monitors validation RMSE epoch-by-epoch. In our experiment, training automatically halted after **Epoch 4** when validation error reached its global minimum (`0.5356`), and optimal model weights were restored.
* **Iterative K-Core Filtering**: Users and recipes with fewer than 10 interactions were filtered out, eliminating isolated 1-rating anomalies.
* **Zero-Data-Leakage Stratified Split**: 70% Train (90,000 interactions), 15% Validation (17,466 interactions), and 15% Test (23,694 interactions) split chronologically per user.

### SVD Training Convergence & Generalization Evidence

| Epoch | Train Loss | Train RMSE | Validation RMSE | Status |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3438 | 0.5860 | 0.5455 | Iterating |
| 2 | 0.3118 | 0.5580 | 0.5388 | Improving |
| 3 | 0.2980 | 0.5455 | 0.5362 | Improving |
| **4** | **0.2898** | **0.5379** | **0.5356** | **Optimal Checkpoint (Restored)** |
| 5 | 0.2843 | 0.5328 | 0.5357 | Plateau |
| 6 | 0.2802 | 0.5289 | 0.5361 | Plateau |
| 7 | 0.2771 | 0.5259 | 0.5363 | Plateau |
| 8 | 0.2745 | 0.5234 | 0.5364 | Early Stop Triggered |

*Notice that while Train RMSE continued to drop from 0.5379 to 0.5234, the Early Stopping controller recognized that Validation RMSE bottomed out at 0.5356 and restored Epoch 4, preventing overparameterized memorization.*

---

## 📊 Empirical Benchmark Results & Model Selection

Evaluated on 23,694 held-out test ratings across active evaluation users:

| Metric | Item-Based CF | SVD (Matrix Factorization) | Content-Based (TF-IDF) | Adaptive Hybrid (Champion) |
| :--- | :---: | :---: | :---: | :---: |
| **NDCG@5** | 0.0025 | 0.0044 | 0.0047 | **0.0047** |
| **NDCG@10** | 0.0036 | 0.0053 | 0.0051 | **0.0071** 🏆 |
| **NDCG@20** | 0.0070 | 0.0075 | 0.0058 | **0.0093** 🏆 |
| **Recall@5** | 0.0037 | 0.0035 | 0.0034 | **0.0048** 🏆 |
| **Recall@10** | 0.0064 | 0.0057 | 0.0055 | **0.0095** 🏆 |
| **Recall@20** | **0.0162** | 0.0126 | 0.0083 | 0.0159 |
| **Precision@10** | 0.0020 | 0.0028 | 0.0030 | **0.0043** 🏆 |
| **MAP@10** | 0.0014 | 0.0025 | 0.0022 | **0.0032** 🏆 |
| **Hit Rate@10** | 2.0% | 2.75% | 2.75% | **4.00%** 🏆 |
| **RMSE (Rating)** | 0.6671 | **0.5667** 🏆 | 2.9877 | 0.8349 |
| **MAE (Rating)** | 0.3777 | **0.3678** 🏆 | 2.9214 | 0.7577 |
| **Catalog Coverage** | 15.29% | 10.41% | **44.25%** 🏆 | 10.31% |

### Key Experimental Insights:
1. **Core Algorithm Comparison (Abstract Goal)**:
   * **SVD Matrix Factorization** outperforms Item-Based CF in **NDCG@10** (0.0053 vs 0.0036), **Precision@10** (0.0028 vs 0.0020), and **RMSE** (0.5667 vs 0.6671).
   * **Item-Based CF** provides strong neighborhood recall on broad catalogs (Recall@20: 0.0162).
2. **Champion Selection**:
   * The **Adaptive Hybrid Recommender** achieves the highest overall ranking performance (**NDCG@10: 0.0071**, **Recall@10: 0.0095**, **Hit Rate@10: 4.00%**), combining latent preferences, item neighborhood signals, and content tokens.

---

## 🗂️ Project Directory Structure

```
special project 2/
├── configs/
│   └── config.yaml                     # Hyperparameters & data paths
├── data/
│   ├── raw/
│   │   ├── df_filtered.pkl             # Food.com parsed metadata & interactions
│   │   └── interactions.parquet        # Raw user interactions (698K records)
│   ├── processed/
│   │   ├── recipes_clean.parquet       # Cleaned recipe catalog (4,179 recipes)
│   │   ├── interactions_clean.parquet  # Cleaned k-core interactions (131K records)
│   │   ├── train.parquet               # Train split (90,000 interactions)
│   │   ├── val.parquet                 # Validation split (17,466 interactions)
│   │   └── test.parquet                # Held-out Test split (23,694 interactions)
│   └── recbole_atomic/                 # RUCAIBox RecBole Atomic Files
│       ├── Food.inter                  # user_id:token, recipe_id:token, rating:float, timestamp:float
│       └── Food.item                   # recipe_id:token, name:token_seq, category:token, calories:float
├── models_saved/
│   ├── item_cf.joblib                  # Serialized Item-Based CF model
│   ├── svd_model.joblib                # Serialized Regularized SVD model
│   ├── content_model.joblib            # Serialized TF-IDF Content model
│   ├── hybrid_model.joblib             # Serialized Adaptive Hybrid model
│   ├── benchmark_results.json          # Complete empirical evaluation report
│   └── diagnostics.json                # SVD training loss & validation RMSE curves
├── src/
│   ├── config.py                       # Configuration loader
│   ├── data/
│   │   ├── downloader.py               # Data fetcher & cache manager
│   │   ├── preprocessor.py             # Feature extraction & RecBole atomic exporter
│   │   └── splitter.py                 # Stratified train/val/test splitting
│   ├── models/
│   │   ├── base.py                     # Base recommender abstract class
│   │   ├── item_cf.py                  # Item-Based CF with shrinkage
│   │   ├── svd_recommender.py          # Biased Regularized SVD with early stopping
│   │   ├── content_based.py            # TF-IDF Vectorizer on recipe text
│   │   ├── knowledge_filter.py         # Rule-based dietary constraint filter
│   │   └── hybrid_recommender.py       # Adaptive Hybrid with online learning
│   ├── evaluation/
│   │   ├── metrics.py                  # Precision, Recall, NDCG, MAP, HitRate, RMSE
│   │   └── evaluator.py                # Comparative benchmark engine
│   ├── api/
│   │   ├── schemas.py                  # Pydantic request/response schemas
│   │   └── main.py                     # FastAPI REST API service
│   ├── frontend/
│   │   └── app.py                      # Streamlit interactive dashboard
│   └── utils/
│       └── logger.py                   # Structured logger
├── tests/
│   ├── test_data.py                    # Preprocessing & RecBole export unit tests
│   ├── test_models.py                  # Model fit/predict/adapt unit tests
│   ├── test_metrics.py                 # Evaluation metrics unit tests
│   └── test_api.py                     # FastAPI endpoint integration tests
├── scripts/
│   └── run_pipeline.py                 # End-to-end pipeline execution script
├── pytest.ini                          # Pytest configuration
├── requirements.txt                    # Python dependencies
├── run_demo.bat                        # Windows interactive runner
└── README.md                           # Comprehensive documentation
```

---

## 🚀 Execution & Usage Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Training & Benchmark Pipeline
To preprocess raw data, train all 4 recommendation algorithms, run the comparative test benchmark, and export models & diagnostics:
```bash
python scripts/run_pipeline.py
```

### 3. Launch the Streamlit Interactive Dashboard
Run the interactive dashboard featuring recipe cards, real-time adaptation simulation, comparative benchmark charts, and overfitting curves:
```bash
streamlit run src/frontend/app.py
```
*Access in browser at:* `http://localhost:8501`

### 4. Launch the FastAPI Production REST API
Start the high-performance async REST API:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger API Docs available at:* `http://127.0.0.1:8000/docs`

### 5. Run the Automated Test Suite
Execute the 21 unit and integration tests across data processing, models, metrics, and API endpoints:
```bash
python -m pytest tests/ -v
```

---

## 📡 REST API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check service health and loaded model status |
| `POST` | `/api/v1/recommend` | Generate top-K recommendations with dietary filters |
| `POST` | `/api/v1/similar` | Find similar recipes (collaborative or content) |
| `POST` | `/api/v1/feedback` | Real-time adaptive feedback submission (online SGD update) |
| `GET` | `/api/v1/benchmark` | Retrieve comparative benchmark metrics table |
| `GET` | `/api/v1/diagnostics`| Retrieve epoch-by-epoch loss and validation RMSE curves |
| `GET` | `/api/v1/recipes/search` | Search recipe catalog by query and dietary tags |
| `GET` | `/api/v1/recipes/{id}` | Get full recipe ingredients and cooking directions |

### Sample Recommendation Request:
```json
POST /api/v1/recommend
{
  "user_id": 2046,
  "top_k": 5,
  "model_type": "hybrid",
  "dietary_constraints": {
    "is_vegetarian": true,
    "max_minutes": 45
  }
}
```

---

## 📜 Citations & References
```bibtex
@inproceedings{majumder-etal-2019-generating,
    title = "Generating Personalized Recipes from Historical User Preferences",
    author = "Majumder, Bodhisattwa Prasad and Li, Shuyang and Ni, Jianmo and McAuley, Julian",
    booktitle = "EMNLP",
    year = "2019",
    pages = "5975--5981",
    url = "https://www.aclweb.org/anthology/D19-1613"
}

@article{recbole2020,
    title = "RecBole: Towards a Unified, Comprehensive and Efficient Recommendation Library",
    author = "Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen",
    journal = "CIKM",
    year = "2021"
}
```
