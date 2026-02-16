"""
NewsMatcher: Cross-lingual semantic search tool.
Author: Nicolas Grivelet

This module leverages Sentence-Transformers to find contextually similar news 
articles across different languages without explicit translation.
"""

import logging
from typing import Any, List, Dict
from sentence_transformers import SentenceTransformer, util

# --- Logging Configuration ---
# Standardized across the project for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NewsMatcher")


class NewsMatcher:
    """
    A semantic search engine to find contextually similar news articles.
    
    It maps articles from different languages into a shared vector space,
    allowing for direct semantic proximity comparison.
    """

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the NewsMatcher with a specific Sentence Transformer model.

        Args:
            model_name (str): The name of the model to load.
                              Defaults to 'paraphrase-multilingual-MiniLM-L12-v2'.
        
        Why this model?
        The 'paraphrase-multilingual-MiniLM-L12-v2' is an excellent trade-off:
        1. Multilingual Support: Handles 50+ languages (ideal for international news).
        2. Performance: High-quality sentence embeddings for semantic similarity.
        3. Efficiency: Lightweight MiniLM architecture optimized for CPU usage.
        """
        logger.info(f"Initializing NewsMatcher with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Semantic engine loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def encode_articles(self, text_list: List[str]) -> Any:
        """
        Transform a list of text articles into dense vector embeddings.

        Args:
            text_list (List[str]): Articles or headlines to vectorize.

        Returns:
            Any: A tensor containing the mathematical representation of the texts.
        """
        logger.info(f"Encoding {len(text_list)} articles into vector space...")
        # convert_to_tensor=True allows for faster similarity calculations
        embeddings = self.model.encode(text_list, convert_to_tensor=True)
        logger.info("Encoding complete.")
        return embeddings

    def rank_matches(self, query_text: str, candidate_list: List[str]) -> List[Dict[str, Any]]:
        """
        Rank candidate articles based on their semantic proximity to the query.

        Args:
            query_text (str): The source article or headline.
            candidate_list (List[str]): Candidates to compare against.

        Returns:
            List[Dict[str, Any]]: Matches sorted by proximity score (descending).
        """
        logger.info(f"Ranking {len(candidate_list)} candidates against the query.")
        
        # 1. Vector-based relevance: Encode query and candidates
        query_embedding = self.encode_articles([query_text])
        candidate_embeddings = self.encode_articles(candidate_list)

        # 2. Compute Semantic Proximity (Cosine Similarity)
        # util.cos_sim returns a matrix [QueryCount x CandidateCount]
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        # 3. Structure and Sort the results
        results = []
        for i, score in enumerate(cosine_scores):
            results.append({
                "candidate_text": candidate_list[i],
                "score": float(score)
            })

        # Higher score means higher semantic similarity
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        logger.info("Semantic matching process completed.")
        return sorted_results


# --- Standalone Demonstration ---
if __name__ == "__main__":
    # Initialize the engine
    matcher = NewsMatcher()

    # 1. Source Article (French) - Topic: Ecological Transition
    source_article_fr = (
        "La France accélère sa transition écologique en investissant massivement "
        "dans les énergies renouvelables pour réduire son empreinte carbone."
    )

    # 2. Candidate Articles (Multilingual)
    candidates = [
        # English: Perfect Match (Renewable energy)
        "France accelerates its ecological transition by investing heavily.",
        
        # Spanish: High Match (Climate action)
        "El gobierno aprueba nuevas leyes para combatir el cambio climático y energías limpias.",
        
        # German: Zero Match (Sports)
        "Bayern München gewinnt das entscheidende Spiel in der Bundesliga mit 3:0.",
        
        # English: Unrelated (Tech/AI)
        "New artificial intelligence models outperform humans in complex strategy games.",
        
        # English: Related (Electric vehicles)
        "Sales of electric vehicles have hit a record high this year as consumers shift."
    ]

    # Run the matching process
    print("\n" + "="*60)
    print(f"SEMANTIC SEARCH RESULTS FOR:\n'{source_article_fr}'")
    print("="*60)
    
    matches = matcher.rank_matches(source_article_fr, candidates)

    for rank, match in enumerate(matches, 1):
        print(f"Rank {rank} | Proximity Score: {match['score']:.4f}")
        print(f"Content: {match['candidate_text']}")
        print("-" * 60)