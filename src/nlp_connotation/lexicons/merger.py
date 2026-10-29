"""
High-performance vectorized lexicon merger for NRC-VAD and MEmoLon datasets.
Replaces O(N*M) nested row-by-row iteration with vectorized O(N) hash joins.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from nlp_connotation.core.exceptions import LexiconError
from nlp_connotation.core.logging import setup_logger

logger = setup_logger(__name__)


class LexiconMerger:
    """
    Merges NRC-VAD lexicon (Valence, Arousal, Dominance) with MEmoLon (Joy, Anger, Sadness, Fear, Disgust)
    using vectorized operations.
    """

    EXPECTED_NRC_COLS = ["English Word", "Valence", "Arousal", "Dominance", "Turkish Word"]
    EXPECTED_MEMOLON_COLS = ["word", "joy", "anger", "sadness", "fear", "disgust"]

    @classmethod
    def merge_datasets(
        cls,
        nrc_vad_path: Path | str,
        memolon_path: Path | str,
        output_csv_path: Optional[Path | str] = None,
        sep_nrc: str = ";",
        sep_memolon: str = "\t",
    ) -> pd.DataFrame:
        """
        Loads NRC-VAD and MEmoLon datasets, validates schema, performs vectorized inner/left join,
        normalizes strings, and optionally saves the output CSV.
        """
        nrc_p = Path(nrc_vad_path)
        memolon_p = Path(memolon_path)

        if not nrc_p.exists():
            raise LexiconError(
                f"NRC-VAD lexicon file not found: {nrc_p}",
                details={"path": str(nrc_p)}
            )
        if not memolon_p.exists():
            raise LexiconError(
                f"MEmoLon dataset file not found: {memolon_p}",
                details={"path": str(memolon_p)}
            )

        logger.info(f"Loading NRC-VAD dataset from {nrc_p}...")
        df_nrc = pd.read_csv(nrc_p, sep=sep_nrc)

        # Validate columns
        missing_nrc = [col for col in cls.EXPECTED_NRC_COLS if col not in df_nrc.columns]
        if missing_nrc:
            raise LexiconError(
                f"NRC-VAD dataset missing expected columns: {missing_nrc}",
                details={"available_columns": df_nrc.columns.tolist()}
            )

        df_nrc = df_nrc[cls.EXPECTED_NRC_COLS].dropna().copy()
        df_nrc["English Word"] = df_nrc["English Word"].astype(str).str.strip().str.lower()
        df_nrc["Turkish Word"] = df_nrc["Turkish Word"].astype(str).str.strip().str.lower()

        logger.info(f"Loading MEmoLon dataset from {memolon_p}...")
        df_memolon = pd.read_csv(memolon_p, sep=sep_memolon)
        
        # Normalize column names in memolon (lowercase)
        df_memolon.columns = [c.lower() for c in df_memolon.columns]
        missing_memolon = [col for col in cls.EXPECTED_MEMOLON_COLS if col not in df_memolon.columns]
        if missing_memolon:
            raise LexiconError(
                f"MEmoLon dataset missing expected columns: {missing_memolon}",
                details={"available_columns": df_memolon.columns.tolist()}
            )

        df_memolon["word"] = df_memolon["word"].astype(str).str.strip().str.lower()
        df_memolon_subset = df_memolon[cls.EXPECTED_MEMOLON_COLS].copy()

        logger.info("Executing vectorized hash-merge on 'English Word' == 'word'...")
        # Vectorized left join - instant execution (replaces O(N*M) nested loop)
        merged = pd.merge(
            df_nrc,
            df_memolon_subset,
            left_on="English Word",
            right_on="word",
            how="left"
        )

        # Drop duplicate join column
        merged.drop(columns=["word"], inplace=True)

        # Rename columns to standard PascalCase
        merged.rename(
            columns={
                "joy": "Joy",
                "anger": "Anger",
                "sadness": "Sadness",
                "fear": "Fear",
                "disgust": "Disgust",
            },
            inplace=True
        )

        # Drop duplicate mappings if any, keeping first
        merged.drop_duplicates(subset=["Turkish Word", "English Word"], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info(f"Merged lexicon generated with {len(merged):,} entries.")

        if output_csv_path:
            out_p = Path(output_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out_p, index=False, encoding="utf-8")
            logger.info(f"Saved merged lexicon to {out_p}")

        return merged
