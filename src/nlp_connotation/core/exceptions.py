"""
Domain exception hierarchy for nlp_connotation.
Guarantees explicit error propagation and prevents silent failures.
"""

from typing import Optional, Any


class NLPConnotationError(Exception):
    """Base exception for all nlp_connotation package errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Context: {self.details}"
        return self.message


class ConfigurationError(NLPConnotationError):
    """Raised when configuration validation or file parsing fails."""
    pass


class PreprocessingError(NLPConnotationError):
    """Raised during text extraction, tokenization, or corpus streaming."""
    pass


class LexiconError(NLPConnotationError):
    """Raised when lexicon loading, merging, or normalization fails."""
    pass


class WordNotFoundError(LexiconError):
    """Raised when a query word does not exist in the lexicon vocabulary."""
    pass


class ModelError(NLPConnotationError):
    """Raised during model loading, initialization, or training."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model artifact or API model cannot be loaded."""
    pass


class EvaluationError(NLPConnotationError):
    """Raised when metric computation or neighborhood extraction fails."""
    pass
