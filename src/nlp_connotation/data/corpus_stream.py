"""
Memory-efficient streaming corpus processor and Wikipedia dump parser.
Guarantees constant memory consumption regardless of dump size.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Callable
import bz2
import os
import xml.etree.ElementTree as ET

from nlp_connotation.core.exceptions import PreprocessingError
from nlp_connotation.core.logging import setup_logger
from nlp_connotation.data.tokenizer import TurkishTokenizer

logger = setup_logger(__name__)


class CorpusSentenceIterator:
    """
    Memory-safe sentence iterator for training Word2Vec models.
    Reads a preprocessed text corpus line by line without loading the full file into RAM.
    Supports repeated iteration across multiple training epochs.
    """

    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise PreprocessingError(
                f"Corpus file does not exist: {self.file_path}",
                details={"path": str(self.file_path)}
            )

    def __iter__(self) -> Iterator[List[str]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


class WikiCorpusExtractor:
    """
    Extracts and tokenizes Wikipedia dump articles (.xml.bz2) in streaming fashion.
    """

    def __init__(
        self,
        tokenizer: Optional[TurkishTokenizer] = None,
        min_article_words: int = 10,
    ):
        self.tokenizer = tokenizer or TurkishTokenizer()
        self.min_article_words = min_article_words

    def extract_and_save(
        self,
        dump_path: Path | str,
        output_path: Path | str,
        batch_log_interval: int = 10000,
        max_articles: Optional[int] = None,
    ) -> int:
        """
        Processes the dump and writes tokenized sentences (space-separated) to output_path.
        Returns the total number of articles saved.
        """
        dump_p = Path(dump_path)
        out_p = Path(output_path)

        if not dump_p.exists():
            raise PreprocessingError(
                f"Wikipedia dump archive not found: {dump_p}",
                details={"dump_path": str(dump_p)}
            )

        out_p.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting streaming extraction from {dump_p} -> {out_p}")
        article_count = 0

        # Try gensim's WikiCorpus if gensim is installed, otherwise fallback to native streaming
        try:
            from gensim.corpora import WikiCorpus

            def _gensim_tokenizer(content, token_min_len=2, token_max_len=50, lower=True):
                return self.tokenizer.tokenize(content)

            wiki = WikiCorpus(str(dump_p), tokenizer_func=_gensim_tokenizer)
            with open(out_p, "w", encoding="utf-8") as out_f:
                for text_tokens in wiki.get_texts():
                    if len(text_tokens) >= self.min_article_words:
                        out_f.write(" ".join(text_tokens) + "\n")
                        article_count += 1
                        if article_count % batch_log_interval == 0:
                            logger.info(f"Processed {article_count:,} articles...")
                        if max_articles and article_count >= max_articles:
                            break

        except ImportError:
            logger.warning("Gensim is not installed. Using native bz2 XML streaming parser.")
            article_count = self._native_extract(dump_p, out_p, batch_log_interval, max_articles)

        logger.info(f"Completed extraction. Total articles processed: {article_count:,}")
        return article_count

    def _native_extract(
        self,
        dump_path: Path,
        output_path: Path,
        batch_log_interval: int,
        max_articles: Optional[int],
    ) -> int:
        """Native bz2 XML iterparse fallback."""
        article_count = 0
        with bz2.open(dump_path, "rt", encoding="utf-8", errors="ignore") as bz_in, \
             open(output_path, "w", encoding="utf-8") as out_f:
            
            # Streaming XML element parsing
            current_text = []
            in_text_tag = False
            for line in bz_in:
                if "<text" in line:
                    in_text_tag = True
                    current_text = [line]
                elif "</text>" in line:
                    in_text_tag = False
                    current_text.append(line)
                    raw_content = "".join(current_text)
                    tokens = self.tokenizer.tokenize(raw_content)
                    if len(tokens) >= self.min_article_words:
                        out_f.write(" ".join(tokens) + "\n")
                        article_count += 1
                        if article_count % batch_log_interval == 0:
                            logger.info(f"Processed {article_count:,} articles...")
                        if max_articles and article_count >= max_articles:
                            break
                    current_text = []
                elif in_text_tag:
                    current_text.append(line)

        return article_count
