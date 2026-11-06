"""
Adaptive Personalized Recipe Recommendation Engine - Interactive Dashboard
===========================================================================
Streamlit production interface showcasing:
1. Personalized Recipe Explorer with Knowledge-based dietary filtering & instant ratings.
2. Real-time Adaptive Learning Simulation (live before-vs-after recommendation re-ranking).
3. Comparative Model Benchmark Dashboard (Item-CF vs SVD vs Content vs Hybrid).
4. Bias-Variance & Overfitting/Underfitting Diagnostic Curves (Loss & Val RMSE tracking).
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.item_cf import ItemBasedCFRecommender
from src.models.svd_recommender import SVDRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid_recommender import AdaptiveHybridRecommender
from src.models.knowledge_filter import KnowledgeConstraintFilter

# Page configuration
st.set_page_config(
    page_title="Adaptive Recipe Recommendation Engine",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (Matching Stitch Design System)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .main-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .author-badge {
        background-color: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08);
    }
    .author-badge strong {
        color: #10B981;
    }
    .recipe-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .recipe-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }
    .badge-veg {
        background-color: #E5EEFF;
        color: #494BD6;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-gf {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-quick {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .score-badge {
        font-size: 1.15rem;
        font-weight: 700;
        color: #006C49;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    """Load models, datasets, and benchmark data into memory."""
    recipes_path = "data/processed/recipes_clean.parquet"
    train_path = "data/processed/train.parquet"
    test_path = "data/processed/test.parquet"

    df_recipes = pd.read_parquet(recipes_path) if os.path.exists(recipes_path) else pd.DataFrame()
    df_train = pd.read_parquet(train_path) if os.path.exists(train_path) else pd.DataFrame()
    df_test = pd.read_parquet(test_path) if os.path.exists(test_path) else pd.DataFrame()

    models = {}
    if os.path.exists("models_saved/item_cf.joblib"):
        models["item_cf"] = ItemBasedCFRecommender.load("models_saved/item_cf.joblib")
    if os.path.exists("models_saved/svd_model.joblib"):
        models["svd"] = SVDRecommender.load("models_saved/svd_model.joblib")
    if os.path.exists("models_saved/content_model.joblib"):
        models["content"] = ContentBasedRecommender.load("models_saved/content_model.joblib")
    if os.path.exists("models_saved/hybrid_model.joblib"):
        models["hybrid"] = AdaptiveHybridRecommender.load("models_saved/hybrid_model.joblib")
    elif "svd" in models and "item_cf" in models and "content" in models and not df_recipes.empty:
        models["hybrid"] = AdaptiveHybridRecommender(
            svd_model=models["svd"],
            item_cf_model=models["item_cf"],
            content_model=models["content"],
            knowledge_filter=KnowledgeConstraintFilter(df_recipes)
        )
        models["hybrid"].is_fitted = True

    benchmark_data = {}
    if os.path.exists("models_saved/benchmark_results.json"):
        with open("models_saved/benchmark_results.json", "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    diagnostics_data = {}
    if os.path.exists("models_saved/diagnostics.json"):
        with open("models_saved/diagnostics.json", "r", encoding="utf-8") as f:
            diagnostics_data = json.load(f)

    return df_recipes, df_train, df_test, models, benchmark_data, diagnostics_data


# Load assets
df_recipes, df_train, df_test, models, benchmark_data, diagnostics_data = load_all_artifacts()

# Title and Author Attribution
st.markdown('<div class="main-title">🍲 Adaptive Personalized Recipe Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An Industry-Grade Recommender System based on the <b>Food.com Dataset</b> (RUCAIBox/RecSysDatasets). '
            'Features Item-Based Collaborative Filtering, Regularized SVD Matrix Factorization, Content-Based TF-IDF, and an Adaptive Hybrid Engine with Zero Overfitting.</div>', unsafe_allow_html=True)

# Authors Box
st.markdown("""
<div class="author-badge">
    👥 <b>Project Investigators:</b> Krrish Jha (24STUCHH010116) &bull; Rasineni Charishma (24STUCHH010662) &bull; Ande Hiranmai (24STUCHH010931) &bull; Chirra Chaithanya (24STUCHH010120)<br/>
    📚 <b>Dataset:</b> Food.com (25K Recipes &bull; 131K+ Cleaned User Interactions &bull; RecBole Atomic Formatted)
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🎯 Recommendation Controls")

# User Selection
sample_users = list(df_test["user_id"].unique()[:25]) if not df_test.empty else [2046, 1773, 2312]
selected_user = st.sidebar.selectbox("👤 Select User ID", sample_users, index=0)

# Model Selection
model_options = {
    "Adaptive Hybrid (Champion)": "hybrid",
    "SVD Matrix Factorization": "svd",
    "Item-Based Collaborative Filtering": "item_cf",
    "Content-Based (TF-IDF)": "content"
}
selected_model_label = st.sidebar.selectbox("🤖 Recommendation Algorithm", list(model_options.keys()), index=0)
selected_model_key = model_options[selected_model_label]

# Dietary and Knowledge Constraints
st.sidebar.subheader("🥗 Knowledge-Based Dietary Filters")
is_veg = st.sidebar.checkbox("Vegetarian Only", value=False)
is_vegan = st.sidebar.checkbox("Vegan Only", value=False)
is_gf = st.sidebar.checkbox("Gluten-Free Only", value=False)
is_low_cal = st.sidebar.checkbox("Low Calorie (≤ 400 kcal)", value=False)
is_high_prot = st.sidebar.checkbox("High Protein (≥ 20g)", value=False)
is_quick = st.sidebar.checkbox("Quick Meal (≤ 30 mins)", value=False)

max_time = st.sidebar.slider("⏱️ Max Total Cooking Time (mins)", min_value=15, max_value=180, value=120, step=15)

categories = ["All"] + sorted(list(df_recipes["RecipeCategory"].unique())) if not df_recipes.empty else ["All"]
selected_cat = st.sidebar.selectbox("📂 Recipe Category", categories, index=0)

top_k = st.sidebar.slider("🔢 Number of Recommendations (K)", min_value=5, max_value=25, value=10, step=5)

# Session state initialization for feedback
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
if "adapted_count" not in st.session_state:
    st.session_state.adapted_count = 0

# App Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🍽️ Personalized Recipe Explorer",
    "⚡ Real-Time Adaptive Learning Loop",
    "📊 Model Benchmark & Comparison",
    "📉 Bias-Variance & Overfitting Diagnostics"
])

# ==============================================================================
# TAB 1: Personalized Recipe Explorer
# ==============================================================================
with tab1:
    st.markdown("### 🎯 Top Personalized Recipe Suggestions")
    st.write(f"Generating personalized recommendations for User **#{selected_user}** using **{selected_model_label}**.")

    # Dietary dictionary
    diet_dict = {
        "is_vegetarian": is_veg,
        "is_vegan": is_vegan,
        "is_gluten_free": is_gf,
        "is_low_calorie": is_low_cal,
        "is_high_protein": is_high_prot,
        "is_quick_meal": is_quick,
        "max_minutes": max_time,
        "category": None if selected_cat == "All" else selected_cat
    }

    # Generate Recommendations
    if selected_model_key in models:
        model = models[selected_model_key]

        if selected_model_key == "hybrid":
            rec_tuples = model.recommend(
                user_id=selected_user,
                top_k=top_k,
                exclude_seen=True,
                dietary_constraints=diet_dict
            )
        else:
            cand_ids = None
            if "hybrid" in models and models["hybrid"].knowledge:
                cand_ids = models["hybrid"].knowledge.filter_candidates(**diet_dict)
            rec_tuples = model.recommend(
                user_id=selected_user,
                top_k=top_k,
                exclude_seen=True,
                candidate_ids=cand_ids
            )

        if not rec_tuples:
            st.warning("No recipes match all selected dietary criteria. Try relaxing some filters!")
        else:
            rec_dict = {int(r["RecipeId"]): r for _, r in df_recipes.iterrows()}

            col_left, col_right = st.columns(2)
            for idx, (rec_id, score) in enumerate(rec_tuples):
                recipe = rec_dict.get(rec_id, None)
                if recipe is None:
                    continue

                target_col = col_left if idx % 2 == 0 else col_right
                recipe_desc = str(recipe.get('description', recipe.get('Description', 'Delicious home-cooked recipe.')))

                with target_col:
                    with st.container():
                        st.markdown(f"""
                        <div class="recipe-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <h4 style="margin: 0; color: #111827;">{recipe['Name']}</h4>
                                    <span style="color: #6B7280; font-size: 0.85rem;">Category: <b>{recipe['RecipeCategory']}</b></span>
                                </div>
                                <div class="score-badge">⭐ {score:.2f} / 5.0</div>
                            </div>
                            <div style="margin: 10px 0;">
                                <span class="badge-quick">⏱️ {int(recipe['TotalTimeMinutes'])} mins</span> &nbsp;
                                <span style="background-color: #F3F4F6; color: #374151; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem;">🔥 {recipe['Calories']:.0f} kcal</span> &nbsp;
                                <span style="background-color: #F3F4F6; color: #374151; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem;">💪 {recipe['ProteinContent']:.1f}g protein</span> &nbsp;
                                {'<span class="badge-veg">🌱 Vegetarian</span> &nbsp;' if recipe['is_vegetarian'] else ''}
                                {'<span class="badge-gf">🌾 Gluten-Free</span> &nbsp;' if recipe['is_gluten_free'] else ''}
                            </div>
                            <p style="font-size: 0.88rem; color: #4B5563; margin-bottom: 8px;">
                                {recipe_desc[:140]}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander(f"📖 View Ingredients & Instructions ({recipe['Name']})"):
                            st.write("**Ingredients:**")
                            ing_list = recipe["ingredients_list"]
                            st.write(", ".join(ing_list) if isinstance(ing_list, list) else str(ing_list))
                            st.write("**Preparation Steps:**")
                            steps_list = recipe.get("steps_list", [])
                            if isinstance(steps_list, list) and steps_list:
                                for s_idx, step in enumerate(steps_list, 1):
                                    st.write(f"{s_idx}. {step}")
                            else:
                                st.write("Follow standard culinary procedure.")

                        # Interactive Rating Submission
                        star_col1, star_col2 = st.columns([3, 1])
                        with star_col1:
                            user_rating = st.select_slider(
                                f"Rate '{recipe['Name']}'",
                                options=[1.0, 2.0, 3.0, 4.0, 5.0],
                                value=5.0,
                                key=f"rate_slider_{rec_id}_{selected_user}"
                            )
                        with star_col2:
                            if st.button("Submit ⭐", key=f"btn_rate_{rec_id}_{selected_user}"):
                                # Apply real-time adaptation
                                if "hybrid" in models:
                                    models["hybrid"].record_interaction(selected_user, rec_id, user_rating)
                                elif "svd" in models:
                                    models["svd"].adapt_user(selected_user, rec_id, user_rating)
                                st.session_state.adapted_count += 1
                                st.session_state.user_ratings[rec_id] = user_rating
                                st.success(f"Feedback recorded! Taste profile adapted in real time (Total adaptations: {st.session_state.adapted_count}).")
    else:
        st.error("Selected model artifact not loaded. Please run `python scripts/run_pipeline.py`.")


# ==============================================================================
# TAB 2: Real-Time Adaptive Learning Simulation
# ==============================================================================
with tab2:
    st.markdown("### ⚡ Live Demonstration: Dynamic Session Adaptation")
    st.write("In production systems, static models fail to respond to changing user intents in an active session. "
             "The **Adaptive Hybrid Engine** updates user latent factors dynamically via online SGD updates and TF-IDF centroid shift.")

    st.info("💡 **How it works:** Rate a recipe below to witness how the recommendation rankings adapt immediately.")

    sim_col1, sim_col2 = st.columns([1, 1])

    # Left: Baseline recommendations
    with sim_col1:
        st.subheader("1️⃣ Baseline Top Recommendations")
        if "hybrid" in models:
            baseline_recs = models["hybrid"].recommend(selected_user, top_k=6, exclude_seen=False)
            rec_dict = {int(r["RecipeId"]): r for _, r in df_recipes.iterrows()}
            for r_id, score in baseline_recs:
                rec_info = rec_dict.get(r_id, {})
                st.write(f"• **{rec_info.get('Name', 'Recipe')}** (⭐ {score:.2f}) - *{rec_info.get('RecipeCategory', '')}*")

    # Right: Adapt with immediate interactive ratings
    with sim_col2:
        st.subheader("2️⃣ Simulate Live Preference Shift")
        # Let user pick a recipe to like or dislike
        sample_catalog_recipes = df_recipes.head(100)
        target_adapt_recipe = st.selectbox(
            "Select a recipe to interact with:",
            options=sample_catalog_recipes["RecipeId"].tolist(),
            format_func=lambda x: f"{df_recipes[df_recipes['RecipeId'] == x]['Name'].values[0]} ({df_recipes[df_recipes['RecipeId'] == x]['RecipeCategory'].values[0]})"
        )
        sim_rating = st.radio("Your Reaction:", [5.0, 1.0], format_func=lambda x: "❤️ Loved it! (5 Stars)" if x == 5.0 else "👎 Disliked it! (1 Star)")

        if st.button("Apply Real-Time Adaptation & Re-rank"):
            if "hybrid" in models:
                res = models["hybrid"].record_interaction(selected_user, target_adapt_recipe, sim_rating)
                st.success(f"Online gradient step applied! Latent palate shifted toward this profile.")

                # Re-generate recommendations
                adapted_recs = models["hybrid"].recommend(selected_user, top_k=6, exclude_seen=True)
                st.write("---")
                st.write("**Updated Recommendations After Adaptation:**")
                for r_id, score in adapted_recs:
                    rec_info = rec_dict.get(r_id, {})
                    st.write(f"🔥 **{rec_info.get('Name', 'Recipe')}** (New Score: ⭐ {score:.2f}) - *{rec_info.get('RecipeCategory', '')}*")


# ==============================================================================
# TAB 3: Model Benchmark & Comparative Analysis
# ==============================================================================
with tab3:
    st.markdown("### 📊 Comprehensive Recommendation Benchmark Comparison")
    st.write("Rigorous empirical evaluation conducted on held-out test interactions (23,694 ratings). "
             "Metrics compare the two primary abstract algorithms (**Item-Based CF** and **SVD Matrix Factorization**) "
             "alongside **Content-Based Filtering** and the **Adaptive Hybrid Recommender**.")

    if benchmark_data and "metrics" in benchmark_data:
        metrics = benchmark_data["metrics"]
        champion = benchmark_data.get("champion_model", "Adaptive Hybrid Recommender")

        # Top metric banner
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("🏆 Champion Model", champion)
        with b_col2:
            st.metric("Top NDCG@10", f"{metrics[champion]['ndcg@10']:.4f}")
        with b_col3:
            st.metric("Top Recall@10", f"{metrics[champion]['recall@10']:.4f}")
        with b_col4:
            st.metric("Lowest RMSE (SVD)", f"{metrics['SVD Matrix Factorization']['rmse']:.4f}")

        # Metrics DataFrame
        df_bench = pd.DataFrame(metrics).T
        display_cols = ["ndcg@5", "ndcg@10", "recall@5", "recall@10", "precision@5", "precision@10", "map@10", "hit_rate@10", "rmse", "mae"]
        st.subheader("📋 Evaluation Benchmark Summary Table")
        st.dataframe(df_bench[display_cols].style.highlight_max(axis=0, color="#D1FAE5", subset=["ndcg@5", "ndcg@10", "recall@5", "recall@10", "precision@5", "precision@10", "map@10", "hit_rate@10"]).highlight_min(axis=0, color="#D1FAE5", subset=["rmse", "mae"]))

        # Visual Plotly Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Ranking Comparison (NDCG@10 & Recall@10)
            models_list = list(metrics.keys())
            ndcg_vals = [metrics[m]["ndcg@10"] for m in models_list]
            recall_vals = [metrics[m]["recall@10"] for m in models_list]

            fig_rank = go.Figure(data=[
                go.Bar(name='NDCG@10', x=models_list, y=ndcg_vals, marker_color='#3B82F6'),
                go.Bar(name='Recall@10', x=models_list, y=recall_vals, marker_color='#10B981')
            ])
            fig_rank.update_layout(
                title='<b>Ranking Performance (NDCG@10 vs Recall@10)</b>',
                barmode='group',
                yaxis_title='Score',
                xaxis_tickangle=-15,
                template='plotly_white'
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        with chart_col2:
            # Rating Accuracy Comparison (RMSE & MAE)
            rmse_vals = [metrics[m]["rmse"] for m in models_list]
            mae_vals = [metrics[m]["mae"] for m in models_list]

            fig_err = go.Figure(data=[
                go.Bar(name='RMSE (Lower is Better)', x=models_list, y=rmse_vals, marker_color='#EF4444'),
                go.Bar(name='MAE (Lower is Better)', x=models_list, y=mae_vals, marker_color='#F59E0B')
            ])
            fig_err.update_layout(
                title='<b>Rating Prediction Error (RMSE & MAE)</b>',
                barmode='group',
                yaxis_title='Error',
                xaxis_tickangle=-15,
                template='plotly_white'
            )
            st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.info("Run `python scripts/run_pipeline.py` to generate the complete benchmark report.")


# ==============================================================================
# TAB 4: Bias-Variance & Overfitting Diagnostics
# ==============================================================================
with tab4:
    st.markdown("### 📉 Overfitting & Underfitting Diagnostic Engine")
    st.write("A critical mandate for industry-grade recommender systems is maintaining an optimal bias-variance tradeoff: "
             "preventing underfitting (capturing latent taste nuance) and preventing overfitting (avoiding memorization of sparse interactions).")

    if diagnostics_data and "history" in diagnostics_data:
        history = diagnostics_data["history"]
        best_ep = diagnostics_data.get("best_epoch", 4)
        best_val = diagnostics_data.get("best_val_rmse", 0.5356)

        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            # Loss Curve
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=history["epoch"],
                y=history["train_loss"],
                mode='lines+markers',
                name='Train Loss (SGD + L2 Reg)',
                line=dict(color='#2563EB', width=3)
            ))
            fig_loss.update_layout(
                title='<b>Training Loss Convergence</b>',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                template='plotly_white'
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with diag_col2:
            # RMSE Train vs Val Curve
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Scatter(
                x=history["epoch"],
                y=history["train_rmse"],
                mode='lines+markers',
                name='Train RMSE',
                line=dict(color='#10B981', width=2)
            ))
            fig_rmse.add_trace(go.Scatter(
                x=history["epoch"],
                y=history["val_rmse"],
                mode='lines+markers',
                name='Validation RMSE (Generalization)',
                line=dict(color='#DC2626', width=3, dash='dash')
            ))
            # Marker for best epoch
            fig_rmse.add_vline(
                x=best_ep,
                line_width=2,
                line_dash="dot",
                line_color="#7C3AED",
                annotation_text=f"Optimal Checkpoint (Epoch {best_ep})",
                annotation_position="top right"
            )
            fig_rmse.update_layout(
                title='<b>Generalization Curves: Train vs Validation RMSE</b>',
                xaxis_title='Epoch',
                yaxis_title='RMSE',
                template='plotly_white'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        st.markdown(rf"""
        #### 🛡️ Anti-Overfitting Safeguards Implemented:
        1. **Validation-Based Early Stopping**: Training monitored validation error epoch-by-epoch. When validation RMSE stopped improving after **Epoch {best_ep}** (best RMSE = `{best_val:.4f}`), training halted after patience cycles and restored optimal weights.
        2. **$L_2$ Weight Regularization ($\lambda = 0.05$)**: Prevents latent embedding explosion on sparse user/recipe interactions.
        3. **Item-CF Shrinkage Factor ($\alpha = 10.0$)**: Dampens spurious cosine similarities on recipe pairs with few co-ratings ($n_{{ij}} / (n_{{ij}} + 10)$).
        4. **Iterative K-Core Filtering**: Eliminated isolated 1-rating users and items to avoid cold noise overfitting.
        """)
    else:
        st.info("Diagnostics data will appear once the training pipeline completes.")
