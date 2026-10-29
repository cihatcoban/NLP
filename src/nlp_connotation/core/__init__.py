"""
Core package exports.
"""

from nlp_connotation.core.config import AppConfig
from nlp_connotation.core.exceptions import (
    NLPConnotationError,
    ConfigurationError,
    PreprocessingError,
    LexiconError,
    WordNotFoundError,
    ModelError,
    ModelNotFoundError,
    EvaluationError,
)
from nlp_connotation.core.logging import setup_logger
from nlp_connotation.core.schemas import (
    ConnotationMetrics,
    WordNeighbor,
    TopKEvaluationResult,
)

__all__ = [
    "AppConfig",
    "NLPConnotationError",
    "ConfigurationError",
    "PreprocessingError",
    "LexiconError",
    "WordNotFoundError",
    "ModelError",
    "ModelNotFoundError",
    "EvaluationError",
    "setup_logger",
    "ConnotationMetrics",
    "WordNeighbor",
    "TopKEvaluationResult",
]
