"""
Lexicon Manager with O(1) indexed lookups and in-memory caching.
Supports bilingual queries (Turkish and English) and returns typed domain objects.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from nlp_connotation.core.exceptions import LexiconError, WordNotFoundError
from nlp_connotation.core.logging import setup_logger
from nlp_connotation.core.schemas import ConnotationMetrics
from nlp_connotation.data.tokenizer import TurkishTokenizer

logger = setup_logger(__name__)


class LexiconManager:
    """
    Manages connotation lexicon lookup with O(1) dictionary indexing.
    Caches metric lookups and handles case-insensitive Turkish/English matches.
    """

    def __init__(self, lexicon_csv_path: Path | str):
        self.lexicon_path = Path(lexicon_csv_path)
        self.tokenizer = TurkishTokenizer()
        self._tr_index: Dict[str, ConnotationMetrics] = {}
        self._en_index: Dict[str, ConnotationMetrics] = {}
        self._load_and_index()

    def _load_and_index(self) -> None:
        if not self.lexicon_path.exists():
            raise LexiconError(
                f"Lexicon CSV not found: {self.lexicon_path}",
                details={"path": str(self.lexicon_path)}
            )

        logger.info(f"Loading lexicon into memory from {self.lexicon_path}...")
        df = pd.read_csv(self.lexicon_path)

        # Standardize column headers
        col_map = {c.strip(): c.strip() for c in df.columns}
        df.rename(columns=col_map, inplace=True)

        required = ["English Word", "Turkish Word", "Valence", "Arousal", "Dominance"]
        for col in required:
            if col not in df.columns:
                raise LexiconError(
                    f"Lexicon file missing critical column: {col}",
                    details={"columns": df.columns.tolist()}
                )

        def _safe_float(val) -> Optional[float]:
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        # Populate O(1) lookup tables
        for _, row in df.iterrows():
            en_word = str(row["English Word"]).strip().lower()
            tr_word = self.tokenizer.to_turkish_lower(str(row["Turkish Word"]).strip())

            metrics = ConnotationMetrics(
                valence=_safe_float(row.get("Valence")),
                arousal=_safe_float(row.get("Arousal")),
                dominance=_safe_float(row.get("Dominance")),
                joy=_safe_float(row.get("Joy")),
                anger=_safe_float(row.get("Anger")),
                sadness=_safe_float(row.get("Sadness")),
                fear=_safe_float(row.get("Fear")),
                disgust=_safe_float(row.get("Disgust")),
            )

            if en_word and en_word not in self._en_index:
                self._en_index[en_word] = metrics

            if tr_word and tr_word not in self._tr_index:
                self._tr_index[tr_word] = metrics

        logger.info(
            f"Lexicon indexed successfully. Unique English: {len(self._en_index):,}, "
            f"Unique Turkish: {len(self._tr_index):,}"
        )

    def get_metrics(self, word: str, lang: str = "tr") -> Optional[ConnotationMetrics]:
        """
        Retrieves ConnotationMetrics for a word. Returns None if the word is absent.
        """
        if not word:
            return None

        lang = lang.lower()
        if lang in ["tr", "turkish"]:
            clean_word = self.tokenizer.to_turkish_lower(word.strip())
            return self._tr_index.get(clean_word)
        elif lang in ["en", "english"]:
            clean_word = word.strip().lower()
            return self._en_index.get(clean_word)
        else:
            raise LexiconError(f"Unsupported language code: {lang}. Use 'tr' or 'en'.")

    def contains(self, word: str, lang: str = "tr") -> bool:
        """Checks whether the word exists in the lexicon."""
        return self.get_metrics(word, lang=lang) is not None

    @property
    def vocabulary_size_tr(self) -> int:
        return len(self._tr_index)

    @property
    def vocabulary_size_en(self) -> int:
        return len(self._en_index)
