"""
Model Adapter abstraction and Registry for cross-lingual word embedding models.
Supports both Gensim KeyedVectors and Native Pure-NumPy KeyedVectors (Python 3.13/3.14+).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from nlp_connotation.core.exceptions import ModelError, ModelNotFoundError
from nlp_connotation.core.logging import setup_logger
from nlp_connotation.models.native_w2v import NativeKeyedVectors

logger = setup_logger(__name__)


class BaseModelAdapter(ABC):
    """Abstract interface for word embedding similarity querying."""

    @abstractmethod
    def most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Returns list of (similar_word, similarity_score)."""
        pass

    @abstractmethod
    def contains(self, word: str) -> bool:
        """Returns True if the word is in the model vocabulary."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class GensimKeyedVectorsAdapter(BaseModelAdapter):
    """Adapter wrapping Gensim or Native KeyedVectors instances."""

    def __init__(self, name: str, keyed_vectors: Any):
        self._name = name
        self._kv = keyed_vectors

    @property
    def model_name(self) -> str:
        return self._name

    def contains(self, word: str) -> bool:
        return word in self._kv

    def most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if not self.contains(word):
            raise ModelError(
                f"Word '{word}' not present in vocabulary of model '{self._name}'.",
                details={"model": self._name, "word": word}
            )
        raw_results = self._kv.most_similar(positive=[word], topn=top_n)
        return [(str(w), float(score)) for w, score in raw_results]


class DictionaryModelAdapter(BaseModelAdapter):
    """
    Lightweight deterministic adapter backed by a dictionary of precomputed neighborhoods.
    Ideal for testing, offline evaluation, and zero-dependency reproducibility.
    """

    def __init__(self, name: str, vocabulary_map: Dict[str, List[Tuple[str, float]]]):
        self._name = name
        self._map = {k.lower(): v for k, v in vocabulary_map.items()}

    @property
    def model_name(self) -> str:
        return self._name

    def contains(self, word: str) -> bool:
        return word.lower() in self._map

    def most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        w_clean = word.lower()
        if w_clean not in self._map:
            raise ModelError(
                f"Word '{word}' not found in dictionary adapter for '{self._name}'.",
                details={"model": self._name, "word": word}
            )
        return self._map[w_clean][:top_n]


class ModelRegistry:
    """
    Central registry for managing local and downloadable embedding models.
    Supports lazy loading and in-memory model pooling.
    """

    def __init__(self, cache_dir: Optional[Path | str] = None):
        self.cache_dir = Path(cache_dir or "data/models")
        self._models: Dict[str, BaseModelAdapter] = {}

    def register_adapter(self, name: str, adapter: BaseModelAdapter) -> None:
        self._models[name] = adapter
        logger.info(f"Registered model adapter '{name}'")

    def get_adapter(self, name: str) -> BaseModelAdapter:
        if name not in self._models:
            raise ModelNotFoundError(
                f"Model '{name}' is not registered in ModelRegistry.",
                details={"available_models": list(self._models.keys())}
            )
        return self._models[name]

    def load_local_word2vec_format(
        self,
        name: str,
        file_path: Path | str,
        binary: bool = True
    ) -> BaseModelAdapter:
        """
        Loads a local Word2Vec binary/text format file.
        Uses Gensim if installed, or automatically falls back to NativeKeyedVectors.
        """
        p = Path(file_path)
        if not p.exists():
            raise ModelNotFoundError(f"Model file not found: {p}", details={"path": str(p)})

        logger.info(f"Loading local Word2Vec model '{name}' from {p} (binary={binary})...")
        try:
            from gensim.models import KeyedVectors
            kv = KeyedVectors.load_word2vec_format(str(p), binary=binary)
        except ImportError:
            logger.info("Using NativeKeyedVectors loader (Pure NumPy)...")
            kv = NativeKeyedVectors.load_word2vec_format(p, binary=binary)

        adapter = GensimKeyedVectorsAdapter(name=name, keyed_vectors=kv)
        self.register_adapter(name, adapter)
        return adapter

    def load_gensim_downloader_model(self, name: str, api_identifier: str) -> BaseModelAdapter:
        """Downloads/loads an English model via gensim.downloader API."""
        try:
            import gensim.downloader as api
        except ImportError as e:
            raise ModelError(
                "gensim.downloader requires Gensim (supported on Python 3.9–3.12). "
                "For Python 3.13/3.14+, use `load_local_word2vec_format` with a local .bin/.vec file."
            ) from e

        logger.info(f"Loading Gensim API model '{api_identifier}'...")
        kv = api.load(api_identifier)
        adapter = GensimKeyedVectorsAdapter(name=name, keyed_vectors=kv)
        self.register_adapter(name, adapter)
        return adapter
