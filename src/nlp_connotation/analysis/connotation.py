"""
Connotation analysis engine.
Computes aggregate mean lexical dimensions (VAD and Discrete Emotions) for top-k neighbors,
explicitly tracking and accounting for vocabulary coverage and missing lexicon tokens.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from nlp_connotation.core.exceptions import EvaluationError
from nlp_connotation.core.logging import setup_logger
from nlp_connotation.core.schemas import (
    ConnotationMetrics,
    WordNeighbor,
    TopKEvaluationResult,
)
from nlp_connotation.lexicons.manager import LexiconManager
from nlp_connotation.models.registry import BaseModelAdapter

logger = setup_logger(__name__)


class ConnotationAnalyzer:
    """
    Evaluates semantic neighbors against connotation lexicons (NRC-VAD + MEmoLon)
    and computes dimension averages with missing-token accounting.
    """

    METRIC_NAMES = ["valence", "arousal", "dominance", "joy", "anger", "sadness", "fear", "disgust"]

    def __init__(self, lexicon_manager: LexiconManager):
        self.lexicon = lexicon_manager

    def evaluate_top_k(
        self,
        model: BaseModelAdapter,
        keyword: str,
        k: int,
        language: str = "tr"
    ) -> TopKEvaluationResult:
        """
        Retrieves top-k neighbors for a keyword, resolves lexicon entries,
        computes dimension means for found words, and records found_k.
        """
        raw_similar = model.most_similar(keyword, top_n=k)
        
        annotated_neighbors: List[WordNeighbor] = []
        found_metrics_list: List[ConnotationMetrics] = []

        for word, score in raw_similar:
            metrics = self.lexicon.get_metrics(word, lang=language)
            in_lex = metrics is not None
            neighbor = WordNeighbor(
                word=word,
                similarity=score,
                metrics=metrics,
                in_lexicon=in_lex
            )
            annotated_neighbors.append(neighbor)
            if in_lex:
                found_metrics_list.append(metrics)

        found_k = len(found_metrics_list)
        mean_metrics = self._calculate_mean_metrics(found_metrics_list)

        return TopKEvaluationResult(
            model_name=model.model_name,
            keyword=keyword,
            language=language,
            requested_k=k,
            found_k=found_k,
            neighbors=annotated_neighbors,
            mean_metrics=mean_metrics
        )

    def _calculate_mean_metrics(
        self,
        metrics_list: List[ConnotationMetrics]
    ) -> ConnotationMetrics:
        """Calculates arithmetic mean across a list of ConnotationMetrics, ignoring NaNs."""
        if not metrics_list:
            return ConnotationMetrics()

        means: Dict[str, Optional[float]] = {}
        for m_name in self.METRIC_NAMES:
            values = [
                getattr(m, m_name)
                for m in metrics_list
                if getattr(m, m_name) is not None
            ]
            if values:
                means[m_name] = float(np.mean(values))
            else:
                means[m_name] = None

        return ConnotationMetrics(**means)

    def run_full_suite(
        self,
        models_and_languages: List[Tuple[BaseModelAdapter, str]],
        test_words_dict: Dict[str, Dict[str, str]],
        k_values: List[int],
        output_dir: Optional[Path | str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes evaluation across all models, words, and k thresholds.
        Returns (summary_dataframe, itemized_dataframe).
        """
        summary_rows: List[Dict[str, Any]] = []
        itemized_rows: List[Dict[str, Any]] = []

        logger.info(
            f"Running connotation evaluation suite: {len(models_and_languages)} models, "
            f"{len(test_words_dict)} test keywords, k={k_values}"
        )

        for model, lang in models_and_languages:
            logger.info(f"Evaluating model: {model.model_name} (Language: {lang})")
            for concept_key, lang_map in test_words_dict.items():
                target_word = lang_map.get(lang)
                if not target_word:
                    logger.warning(f"No word mapping for concept '{concept_key}' in '{lang}'. Skipping.")
                    continue

                if not model.contains(target_word):
                    logger.warning(
                        f"Target word '{target_word}' not in model '{model.model_name}' vocabulary. Skipping."
                    )
                    continue

                for k in k_values:
                    res = self.evaluate_top_k(model, target_word, k=k, language=lang)
                    summary_rows.append(res.to_flattened_dict())

                    # Also store itemized row for each neighbor
                    for neighbor in res.neighbors:
                        m = neighbor.metrics or ConnotationMetrics()
                        itemized_rows.append({
                            "model": model.model_name,
                            "concept": concept_key,
                            "seed_word": target_word,
                            "k_level": k,
                            "neighbor_word": neighbor.word,
                            "similarity": neighbor.similarity,
                            "in_lexicon": neighbor.in_lexicon,
                            "valence": m.valence,
                            "arousal": m.arousal,
                            "dominance": m.dominance,
                            "joy": m.joy,
                            "anger": m.anger,
                            "sadness": m.sadness,
                            "fear": m.fear,
                            "disgust": m.disgust,
                        })

        df_summary = pd.DataFrame(summary_rows)
        df_itemized = pd.DataFrame(itemized_rows)

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            summary_path = out_p / "connotation_summary_report.csv"
            itemized_path = out_p / "neighbor_details_report.csv"
            
            df_summary.to_csv(summary_path, index=False, encoding="utf-8")
            df_itemized.to_csv(itemized_path, index=False, encoding="utf-8")
            logger.info(f"Saved evaluation reports to:\n  - {summary_path}\n  - {itemized_path}")

        return df_summary, df_itemized
