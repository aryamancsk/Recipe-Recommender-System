"""
Interactive Command-Line Recipe Recommender
Author: Aryaman Srimali Jain
"""

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender import RecipeRecommender


def main():
    parser = argparse.ArgumentParser(
        description="Smart Recipe Recommender — Find delicious meals from your available pantry ingredients"
    )
    parser.add_argument(
        "--ingredients",
        "-i",
        type=str,
        default="paneer, tomato, onion, capsicum, cumin, butter",
        help="Comma-separated list of ingredients available in your kitchen",
    )
    parser.add_argument(
        "--diet",
        "-d",
        type=str,
        default="Vegetarian",
        help="Diet constraint (e.g., Vegetarian, Non Vegeterian, High Protein, Diabetic Friendly, or All)",
    )
    parser.add_argument(
        "--cuisine",
        "-c",
        type=str,
        default="All",
        help="Cuisine preference (e.g., Indian, Continental, Italian, North Indian, or All)",
    )
    parser.add_argument(
        "--goal",
        "-g",
        type=str,
        default="balanced",
        choices=["balanced", "weight_loss", "weight_gain"],
        help="Nutritional goal (balanced, weight_loss, weight_gain)",
    )
    parser.add_argument(
        "--max_time",
        "-t",
        type=int,
        default=45,
        help="Maximum total cooking time in minutes",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=3,
        help="Number of recommendations to return",
    )

    args = parser.parse_args()

    engine = RecipeRecommender()
    recs = engine.recommend(
        ingredients=args.ingredients,
        diet=args.diet,
        cuisine=args.cuisine,
        course="All",
        max_time_mins=args.max_time,
        diet_goal=args.goal,
        top_k=args.top_k,
    )

    print("\n" + "=" * 70)
    print("  RECIPE RECOMMENDER SYSTEM - SMART PANTRY & NUTRITION ENGINE")
    print("=" * 70)
    print(f"Pantry Input     : {args.ingredients}")
    print(f"Diet Filter      : {args.diet} | Cuisine: {args.cuisine} | Max Time: {args.max_time} mins")
    print(f"Nutritional Goal : {args.goal.replace('_', ' ').title()}")
    print("-" * 70)

    if not recs:
        print("[!] No recipes found matching all strict filters. Try broadening your criteria.")
        return

    for i, r in enumerate(recs, 1):
        print(f"\n[#{i}] {r['name'].upper()}")
        print(f"     Cuisine: {r['cuisine']} | Course: {r['course']} | Diet: {r['diet']}")
        print(f"     Time: {r['total_time_mins']} mins | Servings: {r['servings']} | Rating: {r['rating']} / 5.0")
        print(f"     Matching Score: {r['composite_score']} (Pantry Coverage: {r['match_percentage']}%)")
        print(f"     Matched Items : {', '.join(r['matched_ingredients']) if r['matched_ingredients'] else 'None'}")
        if r['missing_ingredients']:
            print(f"     Missing Items : {', '.join(r['missing_ingredients'])} (to buy)")
        nut = r['nutrition']
        print(f"     Nutrition     : {nut['calories']} kcal | Protein: {nut['protein_g']}g | Carbs: {nut['carbs_g']}g | Fat: {nut['fat_g']}g")
        print(f"     Instructions  : {r['instructions_snippet']}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
