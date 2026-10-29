"""
Semantic similarity analysis and neighbor discovery.
"""

from typing import List, Tuple
from nlp_connotation.core.exceptions import EvaluationError
from nlp_connotation.core.schemas import WordNeighbor
from nlp_connotation.models.registry import BaseModelAdapter


class SimilarityAnalyzer:
    """Discovers semantic neighbors and similarity rankings for target words."""

    def __init__(self, model_adapter: BaseModelAdapter):
        self.model = model_adapter

    def get_top_neighbors(self, word: str, top_k: int = 10) -> List[WordNeighbor]:
        """
        Retrieves top_k nearest neighbors for a word and returns typed WordNeighbor objects.
        """
        if not self.model.contains(word):
            raise EvaluationError(
                f"Seed word '{word}' is not in the vocabulary of model '{self.model.model_name}'.",
                details={"model": self.model.model_name, "word": word}
            )

        raw_neighbors = self.model.most_similar(word, top_n=top_k)
        return [WordNeighbor(word=w, similarity=score) for w, score in raw_neighbors]
