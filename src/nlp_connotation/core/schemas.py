"""
Domain models, data contracts, and typed schemas for the NLP pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class ConnotationMetrics:
    """Represents VAD and discrete emotion scores for a single word or mean aggregated values."""
    valence: Optional[float] = None
    arousal: Optional[float] = None
    dominance: Optional[float] = None
    joy: Optional[float] = None
    anger: Optional[float] = None
    sadness: Optional[float] = None
    fear: Optional[float] = None
    disgust: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "joy": self.joy,
            "anger": self.anger,
            "sadness": self.sadness,
            "fear": self.fear,
            "disgust": self.disgust,
        }


@dataclass
class WordNeighbor:
    """Represents a single similar word returned from an embedding model."""
    word: str
    similarity: float
    metrics: Optional[ConnotationMetrics] = None
    in_lexicon: bool = False


@dataclass
class TopKEvaluationResult:
    """
    Evaluation record for a specific keyword under an embedding model for top-k words.
    Tracks both requested k and actual found k (accounting for missing lexicon entries).
    """
    model_name: str
    keyword: str
    language: str
    requested_k: int
    found_k: int
    neighbors: List[WordNeighbor] = field(default_factory=list)
    mean_metrics: Optional[ConnotationMetrics] = None

    def to_flattened_dict(self) -> Dict[str, Any]:
        """Flattens the result into a tabular dictionary for CSV / DataFrame reporting."""
        base: Dict[str, Any] = {
            "model": self.model_name,
            "keyword": self.keyword,
            "language": self.language,
            "requested_k": self.requested_k,
            "found_k": self.found_k,
            "similar_words": ", ".join(n.word for n in self.neighbors),
            "found_words_in_lexicon": ", ".join(n.word for n in self.neighbors if n.in_lexicon),
        }
        if self.mean_metrics:
            for key, val in self.mean_metrics.to_dict().items():
                base[f"mean_{key}"] = round(val, 4) if val is not None else None
        else:
            for key in ["valence", "arousal", "dominance", "joy", "anger", "sadness", "fear", "disgust"]:
                base[f"mean_{key}"] = None
        return base
