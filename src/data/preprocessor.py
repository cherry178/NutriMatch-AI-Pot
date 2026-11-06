"""
Data Preprocessing Pipeline
===========================
Processes Food.com recipes and user interaction records into clean tables.
Computes nutritional statistics, dietary flags, duration buckets, and combined text tokens.
Applies k-core interaction filtering to prevent matrix sparsity anomalies.
Exports RecBole atomic files (Food.inter, Food.item) and Parquet tables.
"""

import os
import re
import ast
import pickle
import logging
from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MEAT_KEYWORDS = {
    "chicken", "beef", "pork", "lamb", "turkey", "duck", "bacon", "ham", "sausage",
    "salmon", "tuna", "shrimp", "crab", "lobster", "fish", "meat", "anchovy", "halibut",
    "veal", "prosciutto", "pepperoni", "salami", "steak"
}

DAIRY_EGGS_KEYWORDS = {
    "milk", "butter", "cheese", "yogurt", "cream", "egg", "eggs", "ghee", "parmesan",
    "cheddar", "mozzarella", "mayonnaise", "honey", "ricotta", "sour cream"
}

GLUTEN_KEYWORDS = {
    "flour", "wheat", "bread", "pasta", "spaghetti", "macaroni", "noodle", "noodles",
    "barley", "rye", "couscous", "breadcrumb", "breadcrumbs", "all-purpose flour"
}


def safe_parse_list(val: Any) -> List[str]:
    """Safely parse list representation from string or list."""
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if pd.isna(val):
        return []
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    matches = re.findall(r"'([^']*)'", s) or re.findall(r'"([^"]*)"', s)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    cleaned = s.strip("[]{}()").replace("'", "").replace('"', "")
    return [x.strip() for x in cleaned.split(",") if x.strip()]


def parse_r_vector(val: Any) -> List[str]:
    """Alias for safe_parse_list for backward compatibility."""
    return safe_parse_list(val)


def parse_iso_duration(val: Any) -> int:
    """Parse ISO 8601 duration strings like PT25M, PT1H30M into total minutes."""
    if pd.isna(val):
        return 0
    s = str(val).upper()
    total_mins = 0
    h_match = re.search(r"(\d+)H", s)
    if h_match:
        total_mins += int(h_match.group(1)) * 60
    m_match = re.search(r"(\d+)M", s)
    if m_match:
        total_mins += int(m_match.group(1))
    return total_mins


def safe_parse_nutrition(val: Any) -> List[float]:
    """Parse nutrition array [calories, total_fat, sugar, sodium, protein, saturated_fat, carbs]."""
    if isinstance(val, list):
        return [float(x) for x in val]
    if pd.isna(val):
        return [0.0] * 7
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if nums:
        return [float(x) for x in nums[:7]]
    return [0.0] * 7


def derive_dietary_flags(ingredients_list: List[str], tags_list: List[str]) -> Dict[str, bool]:
    """Determine dietary restrictions from ingredients and tags."""
    ing_text = " ".join(ingredients_list).lower()
    tag_text = " ".join(tags_list).lower()
    
    is_explicit_veg = "vegetarian" in tag_text or "vegan" in tag_text
    is_explicit_vegan = "vegan" in tag_text
    is_explicit_gf = "gluten-free" in tag_text or "gluten free" in tag_text
    
    has_meat = any(meat in ing_text for meat in MEAT_KEYWORDS)
    has_dairy_egg = any(dairy in ing_text for dairy in DAIRY_EGGS_KEYWORDS)
    has_gluten = any(gluten in ing_text for gluten in GLUTEN_KEYWORDS)
    
    is_vegetarian = is_explicit_veg or (not has_meat)
    is_vegan = is_explicit_vegan or (is_vegetarian and not has_dairy_egg)
    is_gluten_free = is_explicit_gf or (not has_gluten)
    
    return {
        "is_vegetarian": bool(is_vegetarian),
        "is_vegan": bool(is_vegan),
        "is_gluten_free": bool(is_gluten_free)
    }


def process_dataset(
    filtered_pkl_path: str = "data/raw/df_filtered.pkl",
    interactions_parquet_path: str = "data/raw/interactions.parquet",
    output_dir: str = "data/processed",
    recbole_dir: str = "data/recbole_atomic",
    min_user_interactions: int = 10,
    min_recipe_interactions: int = 10
) -> Dict[str, Any]:
    """End-to-end dataset processing."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(recbole_dir, exist_ok=True)

    logger.info(f"Loading merged dataset from {filtered_pkl_path}...")
    with open(filtered_pkl_path, "rb") as f:
        df_merged = pickle.load(f)

    # Clean Recipe Catalog
    recipe_cols = ["recipe_id", "name", "minutes", "tags", "nutrition", "n_steps", "steps", "description", "ingredients", "n_ingredients"]
    recipes_raw = df_merged[recipe_cols].drop_duplicates(subset=["recipe_id"]).copy()
    recipes_raw.rename(columns={"recipe_id": "RecipeId", "name": "Name", "minutes": "TotalTimeMinutes"}, inplace=True)
    recipes_raw["RecipeId"] = recipes_raw["RecipeId"].astype(int)

    logger.info(f"Unique recipe records: {len(recipes_raw):,}")

    recipes_raw["ingredients_list"] = recipes_raw["ingredients"].apply(safe_parse_list)
    recipes_raw["tags_list"] = recipes_raw["tags"].apply(safe_parse_list)
    recipes_raw["steps_list"] = recipes_raw["steps"].apply(safe_parse_list)
    
    # Parse nutrition
    nutri_parsed = recipes_raw["nutrition"].apply(safe_parse_nutrition)
    recipes_raw["Calories"] = [n[0] if len(n) > 0 else 0.0 for n in nutri_parsed]
    recipes_raw["FatContent"] = [n[1] if len(n) > 1 else 0.0 for n in nutri_parsed]
    recipes_raw["SugarContent"] = [n[2] if len(n) > 2 else 0.0 for n in nutri_parsed]
    recipes_raw["SodiumContent"] = [n[3] if len(n) > 3 else 0.0 for n in nutri_parsed]
    recipes_raw["ProteinContent"] = [n[4] if len(n) > 4 else 0.0 for n in nutri_parsed]
    recipes_raw["CarbohydrateContent"] = [n[6] if len(n) > 6 else 0.0 for n in nutri_parsed]

    # Derive Dietary Flags
    flags = [derive_dietary_flags(ing, tags) for ing, tags in zip(recipes_raw["ingredients_list"], recipes_raw["tags_list"])]
    recipes_raw["is_vegetarian"] = [f["is_vegetarian"] for f in flags]
    recipes_raw["is_vegan"] = [f["is_vegan"] for f in flags]
    recipes_raw["is_gluten_free"] = [f["is_gluten_free"] for f in flags]
    recipes_raw["is_low_calorie"] = recipes_raw["Calories"] <= 400.0
    recipes_raw["is_high_protein"] = recipes_raw["ProteinContent"] >= 20.0
    recipes_raw["is_quick_meal"] = (recipes_raw["TotalTimeMinutes"] > 0) & (recipes_raw["TotalTimeMinutes"] <= 30)

    # Derive Primary Category from tags or name
    def infer_category(tags: List[str]) -> str:
        for t in tags:
            t_low = t.lower()
            if any(c in t_low for c in ["dinner", "lunch", "breakfast", "dessert", "appetizer", "salad", "soup", "beverage", "bread", "main-dish"]):
                return t_low.title()
        return "Main Dish"
    recipes_raw["RecipeCategory"] = recipes_raw["tags_list"].apply(infer_category)

    # Content-Based text token representation
    recipes_raw["content_text"] = (
        recipes_raw["Name"].fillna("") + " " +
        recipes_raw["RecipeCategory"].fillna("") + " " +
        recipes_raw["tags_list"].apply(lambda l: " ".join(l)) + " " +
        recipes_raw["ingredients_list"].apply(lambda l: " ".join(l)) + " " +
        recipes_raw["description"].fillna("")
    ).str.lower().str.replace(r"[^\w\s]", " ", regex=True)

    # Clean Interactions
    logger.info(f"Loading full interaction set from {interactions_parquet_path}...")
    df_inter_full = pd.read_parquet(interactions_parquet_path)
    # Combine with interactions in df_merged
    inter_cols_merged = ["user_id", "recipe_id", "date", "rating"]
    inter_merged = df_merged[inter_cols_merged].copy()
    
    all_inter = pd.concat([df_inter_full[["user_id", "recipe_id", "date", "rating"]], inter_merged], ignore_index=True)
    all_inter.drop_duplicates(subset=["user_id", "recipe_id"], keep="last", inplace=True)
    all_inter["user_id"] = all_inter["user_id"].astype(int)
    all_inter["recipe_id"] = all_inter["recipe_id"].astype(int)
    all_inter["rating"] = all_inter["rating"].astype(float)
    
    # Filter positive/explicit ratings (1 to 5)
    all_inter = all_inter[all_inter["rating"] >= 1.0].copy()
    
    # Restrict to recipes present in our catalog
    valid_recipes = set(recipes_raw["RecipeId"])
    all_inter = all_inter[all_inter["recipe_id"].isin(valid_recipes)].copy()
    logger.info(f"Interactions matching catalog recipes: {len(all_inter):,}")

    # Iterative k-core filtering
    prev_len = len(all_inter)
    iteration = 1
    while True:
        u_counts = all_inter["user_id"].value_counts()
        r_counts = all_inter["recipe_id"].value_counts()

        valid_users = u_counts[u_counts >= min_user_interactions].index
        valid_rec = r_counts[r_counts >= min_recipe_interactions].index

        all_inter = all_inter[
            all_inter["user_id"].isin(valid_users) & all_inter["recipe_id"].isin(valid_rec)
        ].copy()

        curr_len = len(all_inter)
        logger.info(f"k-core iteration {iteration}: {curr_len:,} interactions across {all_inter['user_id'].nunique():,} users and {all_inter['recipe_id'].nunique():,} recipes")
        if curr_len == prev_len or curr_len == 0:
            break
        prev_len = curr_len
        iteration += 1

    # Clean and standardize date column
    all_inter["date"] = pd.to_datetime(all_inter["date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("2020-01-01")

    # Filter recipes down to active ones
    active_recipes = set(all_inter["recipe_id"])
    recipes_clean = recipes_raw[recipes_raw["RecipeId"].isin(active_recipes)].copy()

    # Save Cleaned Parquet
    inter_out = os.path.join(output_dir, "interactions_clean.parquet")
    rec_out = os.path.join(output_dir, "recipes_clean.parquet")
    all_inter.to_parquet(inter_out, index=False)
    recipes_clean.to_parquet(rec_out, index=False)
    logger.info(f"Saved cleaned interactions to: {inter_out}")
    logger.info(f"Saved cleaned recipes to: {rec_out}")

    # Export RecBole Atomic files
    inter_file, item_file = export_recbole_atomic_files(all_inter, recipes_clean, output_dir=recbole_dir)

    return {
        "interactions_path": inter_out,
        "recipes_path": rec_out,
        "atomic_inter": inter_file,
        "atomic_item": item_file,
        "n_users": all_inter["user_id"].nunique(),
        "n_recipes": all_inter["recipe_id"].nunique(),
        "n_interactions": len(all_inter)
    }


def export_recbole_atomic_files(
    interactions_df: pd.DataFrame,
    recipes_df: pd.DataFrame,
    output_dir: str = "data/recbole_atomic"
) -> Tuple[str, str]:
    """Export atomic benchmark files conforming to RUCAIBox RecBole specification."""
    os.makedirs(output_dir, exist_ok=True)
    inter_file = os.path.join(output_dir, "Food.inter")
    item_file = os.path.join(output_dir, "Food.item")

    recbole_inter = interactions_df[["user_id", "recipe_id", "rating"]].copy()
    if "date" in interactions_df.columns:
        recbole_inter["timestamp"] = pd.to_datetime(interactions_df["date"], errors="coerce").fillna(pd.Timestamp("2020-01-01")).astype("int64") // 10**9
    else:
        recbole_inter["timestamp"] = 0.0

    recbole_inter.columns = ["user_id:token", "recipe_id:token", "rating:float", "timestamp:float"]
    recbole_inter.to_csv(inter_file, sep="\t", index=False)

    recbole_item = pd.DataFrame({
        "recipe_id:token": recipes_df["RecipeId"],
        "name:token_seq": recipes_df["Name"].astype(str).str.replace(r"\s+", " ", regex=True),
        "category:token": recipes_df["RecipeCategory"].astype(str).str.replace(" ", "_"),
        "calories:float": recipes_df["Calories"]
    })
    recbole_item.to_csv(item_file, sep="\t", index=False)
    logger.info(f"Exported RecBole atomic files to {output_dir} (Food.inter & Food.item)")
    return inter_file, item_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process_dataset()
