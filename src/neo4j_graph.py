"""
Neo4j Knowledge Graph Pipeline for Recipe Recommender
Author: Aryaman Srimali Jain
"""

from pathlib import Path
from typing import List, Dict, Optional

class RecipeKnowledgeGraph:
    """
    Constructs and queries a Knowledge Graph representation of recipes in Neo4j.
    
    Graph Schema:
      (:Recipe {id, name, cuisine, course, diet, prep_time, rating})
      (:Ingredient {name, idf_weight})
      (:Cuisine {name})
      (:Diet {name})
      
    Relationships:
      (:Recipe)-[:CONTAINS {amount}]->(:Ingredient)
      (:Recipe)-[:BELONGS_TO]->(:Cuisine)
      (:Recipe)-[:SUITABLE_FOR]->(:Diet)
    """

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j instance if running."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            print(f"[OK] Connected to Neo4j Knowledge Graph at {self.uri}")
        except Exception as e:
            print(f"[NOTE] Neo4j server offline ({e}). Using in-memory TF-IDF engine instead.")

    @staticmethod
    def get_recipe_query_cypher(ingredients: List[str], min_overlap_ratio: float = 0.5) -> str:
        """
        Generate Cypher query to retrieve recipes matching ingredient combinations.
        """
        ing_list_str = str(ingredients).lower()
        return f"""
        MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient)
        WHERE toLower(i.name) IN {ing_list_str}
        WITH r, count(i) AS matched_count, collect(i.name) AS matched_ingredients
        MATCH (r)-[:CONTAINS]->(all_i:Ingredient)
        WITH r, matched_count, matched_ingredients, count(all_i) AS total_ingredients
        WHERE (toFloat(matched_count) / toFloat(total_ingredients)) >= {min_overlap_ratio}
        RETURN r.name AS Recipe,
               r.cuisine AS Cuisine,
               r.diet AS Diet,
               r.rating AS Rating,
               matched_count AS MatchedCount,
               total_ingredients AS TotalCount,
               (toFloat(matched_count) / toFloat(total_ingredients)) AS MatchRatio
        ORDER BY MatchRatio DESC, r.rating DESC
        LIMIT 10;
        """

    def close(self):
        if self.driver:
            self.driver.close()
