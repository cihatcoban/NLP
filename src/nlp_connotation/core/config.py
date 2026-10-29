"""
Configuration parser and validator for nlp_connotation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field

from nlp_connotation.core.exceptions import ConfigurationError


class ProjectSettings(BaseModel):
    name: str = "turkish-word2vec-connotation"
    version: str = "1.0.0"
    seed: int = 42


class PathsSettings(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    lexicons_dir: Path = Path("data/lexicons")
    models_dir: Path = Path("data/models")
    results_dir: Path = Path("data/results")
    wiki_dump: Path = Path("data/raw/trwiki-latest-pages-articles.xml.bz2")
    processed_corpus: Path = Path("data/processed/wiki.tr.txt")
    nrc_vad_path: Path = Path("data/raw/Turkish-NRC-VAD-Lexicon.csv")
    memolon_path: Path = Path("data/raw/MTL_grouped/en.tsv")
    merged_lexicon: Path = Path("data/lexicons/my_Lexicon.csv")


class TokenizationSettings(BaseModel):
    min_token_len: int = 2
    max_token_len: int = 50
    lowercase: bool = True
    filter_non_alpha: bool = False


class TrainingSettings(BaseModel):
    vector_size: int = 300
    window: int = 5
    min_count: int = 5
    workers: int = 4
    epochs: int = 5
    sg: int = 1  # 1 for Skip-gram, 0 for CBOW
    binary_format: bool = True
    output_model: Path = Path("data/models/turkish_word2vec.bin")


class WordPair(BaseModel):
    tr: str
    en: str


class EvaluationSettings(BaseModel):
    k_values: List[int] = Field(default_factory=lambda: [1, 3, 10])
    test_words: Dict[str, WordPair] = Field(default_factory=dict)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"


class AppConfig(BaseModel):
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    tokenization: TokenizationSettings = Field(default_factory=TokenizationSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AppConfig":
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigurationError(
                f"Config file not found: {file_path}",
                details={"path": str(file_path)}
            )
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse configuration: {e}",
                details={"path": str(file_path), "error": str(e)}
            ) from e
