"""
Core Recipe Recommender Engine
Author: Aryaman Srimali Jain
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "recipes_master.csv"


class RecipeRecommender:
    """
    Hybrid Recipe Recommender combining:
    1. TF-IDF Inverted Frequency Ingredient Matching
    2. Exact & Partial Ingredient Overlap (Pantry Coverage)
    3. Nutritional Goal Conditioning (Weight Loss / Weight Gain / Balanced)
    4. Multi-Faceted Filters (Diet, Cuisine, Course, Max Time)
    """

    def __init__(self, data_path: Union[str, Path] = DEFAULT_DATA_PATH):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Recipe dataset not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        self.df["ingredients_clean"] = self.df["ingredients_clean"].fillna("")
        self._init_vectorizer()

    def _init_vectorizer(self):
        """Fit TF-IDF matrix on cleaned ingredient strings."""
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=1200)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["ingredients_clean"])

    @staticmethod
    def _clean_input_tokens(ingredients: Union[str, List[str]]) -> List[str]:
        if isinstance(ingredients, str):
            raw_list = ingredients.split(",")
        else:
            raw_list = ingredients

        tokens = []
        for item in raw_list:
            clean = re.sub(r"[^a-zA-Z\s]", " ", item).lower().strip()
            clean = re.sub(r"\s+", " ", clean)
            if clean and len(clean) > 2:
                tokens.append(clean)
        return list(dict.fromkeys(tokens))

    def _compute_nutritional_score(self, sub_df: pd.DataFrame, goal: str) -> np.ndarray:
        """Compute normalized nutritional score based on health objective."""
        cal = sub_df["calories"].values
        protein = sub_df["protein_g"].values
        fat = sub_df["fat_g"].values
        fiber = sub_df["fiber_g"].values
        sugar = sub_df["sugar_g"].values

        # Robust min-max normalization
        def norm(arr):
            span = np.ptp(arr)
            return (arr - np.min(arr)) / (span + 1e-6)

        norm_cal = norm(cal)
        norm_prot = norm(protein)
        norm_fat = norm(fat)
        norm_fiber = norm(fiber)
        norm_sugar = norm(sugar)

        if goal == "weight_loss":
            # Higher protein & fiber, lower calories, fat & sugar
            score = 0.40 * norm_prot + 0.30 * norm_fiber - 0.20 * norm_cal - 0.10 * norm_sugar
        elif goal == "weight_gain":
            # Higher calories, protein, balanced fat
            score = 0.40 * norm_cal + 0.35 * norm_prot + 0.15 * norm_fat + 0.10 * norm_fiber
        else:  # balanced
            # Moderate calories, balanced macros
            score = 0.35 * norm_prot + 0.25 * norm_fiber - 0.20 * np.abs(norm_cal - 0.5) - 0.20 * norm_sugar

        return norm(score)

    def recommend(
        self,
        ingredients: Union[str, List[str]],
        diet: Optional[str] = None,
        cuisine: Optional[str] = None,
        course: Optional[str] = None,
        max_time_mins: Optional[int] = None,
        diet_goal: str = "balanced",
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Recommend top-k recipes matching user pantry items and constraints.
        """
        input_tokens = self._clean_input_tokens(ingredients)
        if not input_tokens:
            return []

        query_str = " ".join(input_tokens)
        query_vec = self.vectorizer.transform([query_str])

        # Step 1: Filter candidates based on categorical constraints
        mask = pd.Series(True, index=self.df.index)
        if diet and diet.lower() != "all":
            mask &= self.df["diet"].str.lower().str.contains(diet.lower())
        if cuisine and cuisine.lower() != "all":
            mask &= self.df["cuisine"].str.lower().str.contains(cuisine.lower())
        if course and course.lower() != "all":
            mask &= self.df["course"].str.lower().str.contains(course.lower())
        if max_time_mins and max_time_mins > 0:
            mask &= self.df["total_time_mins"] <= max_time_mins

        candidate_indices = np.where(mask.values)[0]
        if len(candidate_indices) == 0:
            return []

        # Step 2: Compute TF-IDF Cosine Similarity
        candidate_tfidf = self.tfidf_matrix[candidate_indices]
        sim_scores = cosine_similarity(query_vec, candidate_tfidf).flatten()

        candidate_df = self.df.iloc[candidate_indices].copy()
        candidate_df["tfidf_sim"] = sim_scores

        # Step 3: Compute Overlap & Missing Ingredients
        input_set = set(input_tokens)
        matched_ing_list = []
        missing_ing_list = []
        match_ratios = []

        for row_ing in candidate_df["ingredients_clean"]:
            recipe_ings = [x.strip() for x in row_ing.split(",") if x.strip()]
            recipe_set = set(recipe_ings)

            matched = [r for r in recipe_ings if any(u in r or r in u for u in input_set)]
            missing = [r for r in recipe_ings if not any(u in r or r in u for u in input_set)]

            overlap_ratio = len(matched) / (len(recipe_ings) + 1e-6)
            matched_ing_list.append(matched)
            missing_ing_list.append(missing)
            match_ratios.append(overlap_ratio)

        candidate_df["match_ratio"] = match_ratios
        candidate_df["matched_ingredients"] = matched_ing_list
        candidate_df["missing_ingredients"] = missing_ing_list

        # Step 4: Compute Goal-driven Nutritional Score
        nutri_scores = self._compute_nutritional_score(candidate_df, diet_goal)
        candidate_df["nutri_score"] = nutri_scores

        # Step 5: Composite Ranking Formulation
        # S = 0.50 * TFIDF + 0.25 * Overlap + 0.15 * Nutri + 0.10 * (Rating/5.0)
        norm_rating = candidate_df["rating"].values / 5.0
        final_scores = (
            0.50 * candidate_df["tfidf_sim"].values
            + 0.25 * candidate_df["match_ratio"].values
            + 0.15 * candidate_df["nutri_score"].values
            + 0.10 * norm_rating
        )
        candidate_df["composite_score"] = final_scores

        # Sort descending
        top_candidates = candidate_df.sort_values(by="composite_score", ascending=False).head(top_k)

        results = []
        for _, row in top_candidates.iterrows():
            results.append({
                "recipe_id": int(row["recipe_id"]),
                "name": str(row["name"]),
                "cuisine": str(row["cuisine"]),
                "course": str(row["course"]),
                "diet": str(row["diet"]),
                "total_time_mins": int(row["total_time_mins"]),
                "servings": int(row["servings"]),
                "rating": float(row["rating"]),
                "composite_score": round(float(row["composite_score"]), 3),
                "match_percentage": round(float(row["match_ratio"]) * 100, 1),
                "matched_ingredients": row["matched_ingredients"],
                "missing_ingredients": row["missing_ingredients"][:5],  # top 5 missing
                "nutrition": {
                    "calories": int(row["calories"]),
                    "protein_g": round(float(row["protein_g"]), 1),
                    "carbs_g": round(float(row["carbs_g"]), 1),
                    "fat_g": round(float(row["fat_g"]), 1),
                    "fiber_g": round(float(row["fiber_g"]), 1),
                },
                "instructions_snippet": str(row["instructions"])[:220] + "...",
                "url": str(row["url"]),
            })

        return results
