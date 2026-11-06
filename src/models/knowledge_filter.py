"""
Knowledge-Based Recommendation & Constraint Filter
===================================================
Applies domain knowledge and explicit dietary/nutritional constraints to candidate recipes.
Supports dietary lifestyles (Vegetarian, Vegan, Gluten-Free), preparation limits (Quick-Cook < 30m),
allergen exclusions (peanuts, dairy, seafood), and calorie/macronutrient bounds.
"""

from typing import List, Dict, Optional, Set, Any
import pandas as pd


class KnowledgeConstraintFilter:
    """
    Applies rule-based constraints to candidate recipe catalogs.
    """

    def __init__(self, recipes_df: pd.DataFrame):
        self.recipes_df = recipes_df.copy()
        self.recipe_dict = {int(r["RecipeId"]): r for _, r in self.recipes_df.iterrows()}

    def filter_candidates(
        self,
        candidate_ids: Optional[List[int]] = None,
        is_vegetarian: bool = False,
        is_vegan: bool = False,
        is_gluten_free: bool = False,
        is_low_calorie: bool = False,
        is_high_protein: bool = False,
        is_quick_meal: bool = False,
        max_minutes: Optional[int] = None,
        max_calories: Optional[float] = None,
        min_protein: Optional[float] = None,
        exclude_ingredients: Optional[List[str]] = None,
        include_ingredients: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> List[int]:
        """
        Filter candidate recipe IDs based on user-specified knowledge constraints.
        """
        df = self.recipes_df
        if candidate_ids is not None:
            df = df[df["RecipeId"].isin(set(candidate_ids))]

        if is_vegetarian:
            df = df[df["is_vegetarian"] == True]

        if is_vegan:
            df = df[df["is_vegan"] == True]

        if is_gluten_free:
            df = df[df["is_gluten_free"] == True]

        if is_low_calorie:
            df = df[df["is_low_calorie"] == True]

        if is_high_protein:
            df = df[df["is_high_protein"] == True]

        if is_quick_meal:
            df = df[df["is_quick_meal"] == True]

        if max_minutes is not None and max_minutes > 0:
            df = df[(df["TotalTimeMinutes"] > 0) & (df["TotalTimeMinutes"] <= max_minutes)]

        if max_calories is not None and max_calories > 0:
            df = df[df["Calories"] <= max_calories]

        if min_protein is not None and min_protein > 0:
            df = df[df["ProteinContent"] >= min_protein]

        if category and category.lower() != "all":
            df = df[df["RecipeCategory"].str.lower() == category.lower()]

        if exclude_ingredients:
            excl_lower = [x.lower() for x in exclude_ingredients if x.strip()]
            for item in excl_lower:
                df = df[~df["content_text"].str.contains(item, na=False)]

        if include_ingredients:
            inc_lower = [x.lower() for x in include_ingredients if x.strip()]
            for item in inc_lower:
                df = df[df["content_text"].str.contains(item, na=False)]

        return [int(r) for r in df["RecipeId"].tolist()]
