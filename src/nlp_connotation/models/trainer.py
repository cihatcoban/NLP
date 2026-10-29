"""
Word2Vec Model Trainer with streaming corpus integration and automatic fallback
to Pure-NumPy engine on Python 3.13 / 3.14+ where Gensim C-extensions fail to compile.
"""

from pathlib import Path
from typing import Optional, Any
from multiprocessing import cpu_count

from nlp_connotation.core.exceptions import ModelError
from nlp_connotation.core.logging import setup_logger
from nlp_connotation.core.config import TrainingSettings
from nlp_connotation.data.corpus_stream import CorpusSentenceIterator
from nlp_connotation.models.native_w2v import NativeWord2VecModel

logger = setup_logger(__name__)


class Word2VecTrainer:
    """
    Trains Word2Vec embedding models over streaming text corpora.
    Uses Gensim Cython engine if installed, or automatically falls back to
    NativeWord2VecModel (Pure NumPy) on Python 3.13/3.14+.
    """

    def __init__(self, settings: Optional[TrainingSettings] = None):
        self.settings = settings or TrainingSettings()

    def train(
        self,
        corpus_path: Path | str,
        output_model_path: Optional[Path | str] = None,
    ) -> Any:
        """
        Trains the Word2Vec model on the given corpus path and saves the KeyedVectors.
        """
        corpus_p = Path(corpus_path)
        if not corpus_p.exists():
            raise ModelError(
                f"Corpus file does not exist: {corpus_p}",
                details={"path": str(corpus_p)}
            )

        out_path = Path(output_model_path or self.settings.output_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing streaming sentence iterator for: {corpus_p}")
        sentences = CorpusSentenceIterator(corpus_p)

        try:
            from gensim.models import Word2Vec
            effective_workers = min(self.settings.workers, cpu_count())
            logger.info(
                f"Starting Gensim Word2Vec training: vector_size={self.settings.vector_size}, "
                f"window={self.settings.window}, min_count={self.settings.min_count}, "
                f"epochs={self.settings.epochs}, workers={effective_workers}"
            )
            model = Word2Vec(
                sentences=sentences,
                vector_size=self.settings.vector_size,
                window=self.settings.window,
                min_count=self.settings.min_count,
                workers=effective_workers,
                epochs=self.settings.epochs,
                sg=self.settings.sg,
            )
        except ImportError:
            logger.warning(
                "Gensim is not installed (or running on Python 3.13/3.14+). "
                "Using Native Pure-NumPy Word2Vec Engine."
            )
            model = NativeWord2VecModel(
                sentences=sentences,
                vector_size=self.settings.vector_size,
                window=self.settings.window,
                min_count=self.settings.min_count,
                epochs=self.settings.epochs,
            )

        logger.info(f"Training completed. Vocabulary size: {len(model.wv):,}")

        if self.settings.binary_format:
            logger.info(f"Saving KeyedVectors in binary format to {out_path}...")
            model.wv.save_word2vec_format(str(out_path), binary=True)
        else:
            logger.info(f"Saving model to {out_path}...")
            model.wv.save_word2vec_format(str(out_path), binary=False)

        logger.info("Model successfully saved.")
        return model
