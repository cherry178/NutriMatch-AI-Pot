"""
API Schemas
===========
Pydantic schemas for REST API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DietaryPreferences(BaseModel):
    is_vegetarian: bool = False
    is_vegan: bool = False
    is_gluten_free: bool = False
    is_low_calorie: bool = False
    is_high_protein: bool = False
    is_quick_meal: bool = False
    max_minutes: Optional[int] = None
    max_calories: Optional[float] = None
    min_protein: Optional[float] = None
    category: Optional[str] = None
    exclude_ingredients: Optional[List[str]] = None
    include_ingredients: Optional[List[str]] = None


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user requesting recommendations")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    offset: int = Field(0, ge=0, description="Offset for pagination and loading more dishes")
    model_type: str = Field("hybrid", description="Model to use: 'hybrid', 'svd', 'item_cf', 'content'")
    dietary_constraints: Optional[DietaryPreferences] = None
    exclude_seen: bool = True


class RecipeItem(BaseModel):
    recipe_id: int
    name: str
    category: str
    predicted_score: float
    total_time_minutes: int
    calories: float
    protein_g: float
    is_vegetarian: bool
    is_vegan: bool
    is_gluten_free: bool
    ingredients: List[str]
    description: str
    instructions: List[str] = Field(default_factory=list, description="Ordered step-by-step cooking instructions")
    steps: List[str] = Field(default_factory=list, description="Ordered step-by-step cooking instructions")


class RecommendationResponse(BaseModel):
    user_id: int
    model_used: str
    recommendations: List[RecipeItem]
    count: int


class SimilarRecipesRequest(BaseModel):
    recipe_id: int
    top_k: int = Field(10, ge=1, le=50)
    method: str = Field("collaborative", description="'collaborative' or 'content'")


class FeedbackRequest(BaseModel):
    user_id: int
    recipe_id: int
    rating: float = Field(..., ge=1.0, le=5.0)
    feedback_type: str = "rating"


class FeedbackResponse(BaseModel):
    status: str
    message: str
    user_id: int
    recipe_id: int
    rating: float
    updated_weights: Optional[Dict[str, float]] = None
    delta_score: Optional[float] = None
    re_ranked_recommendations: Optional[List[RecipeItem]] = None


class UserProfileResponse(BaseModel):
    user_id: int
    persona: str
    interaction_count: int
    avg_rating: float
    metrics: Dict[str, float]
    recent_interactions: List[Dict[str, Any]]
    latent_weights: Dict[str, float]

