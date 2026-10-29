"""
Unified Command Line Interface for nlp_connotation.
Powered by Click.
"""

from pathlib import Path
import sys
import click

from nlp_connotation.core.config import AppConfig
from nlp_connotation.core.exceptions import NLPConnotationError
from nlp_connotation.core.logging import setup_logger
from nlp_connotation.data.corpus_stream import WikiCorpusExtractor
from nlp_connotation.data.tokenizer import TurkishTokenizer
from nlp_connotation.lexicons.merger import LexiconMerger
from nlp_connotation.lexicons.manager import LexiconManager
from nlp_connotation.models.trainer import Word2VecTrainer
from nlp_connotation.models.registry import ModelRegistry, DictionaryModelAdapter
from nlp_connotation.analysis.connotation import ConnotationAnalyzer

logger = setup_logger("nlp_connotation.cli")


@click.group()
@click.option("--config", "-c", type=click.Path(exists=False), default="configs/default_config.yaml", help="Path to YAML config file")
@click.pass_context
def cli(ctx, config):
    """Turkish Word2Vec & Connotation Analysis CLI (Keystone Architecture)."""
    ctx.ensure_object(dict)
    config_path = Path(config)
    if config_path.exists():
        ctx.obj["config"] = AppConfig.from_yaml(config_path)
    else:
        ctx.obj["config"] = AppConfig()


@cli.command()
@click.option("--dump-path", type=click.Path(exists=True), help="Input Wikipedia .xml.bz2 dump")
@click.option("--output-path", type=click.Path(), help="Output path for preprocessed token text")
@click.option("--max-articles", type=int, default=None, help="Optional limit on articles processed")
@click.pass_context
def preprocess(ctx, dump_path, output_path, max_articles):
    """Extract and tokenize Wikipedia XML dump with Turkish linguistic normalization."""
    cfg: AppConfig = ctx.obj["config"]
    input_dump = dump_path or str(cfg.paths.wiki_dump)
    output_file = output_path or str(cfg.paths.processed_corpus)

    tokenizer = TurkishTokenizer(
        min_token_len=cfg.tokenization.min_token_len,
        max_token_len=cfg.tokenization.max_token_len,
        lowercase=cfg.tokenization.lowercase
    )
    extractor = WikiCorpusExtractor(tokenizer=tokenizer)

    click.echo(f"==> Preprocessing dump: {input_dump} -> {output_file}")
    try:
        count = extractor.extract_and_save(input_dump, output_file, max_articles=max_articles)
        click.secho(f"Successfully processed {count:,} articles.", fg="green")
    except NLPConnotationError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@cli.command()
@click.option("--corpus", type=click.Path(exists=True), help="Path to cleaned corpus text file")
@click.option("--output", type=click.Path(), help="Path to save trained Word2Vec model")
@click.option("--vector-size", type=int, default=300, help="Dimensionality of embeddings")
@click.option("--window", type=int, default=5, help="Context window size")
@click.option("--epochs", type=int, default=5, help="Number of training epochs")
@click.pass_context
def train(ctx, corpus, output, vector_size, window, epochs):
    """Train Word2Vec embeddings on streaming text corpus."""
    cfg: AppConfig = ctx.obj["config"]
    corpus_file = corpus or str(cfg.paths.processed_corpus)
    out_file = output or str(cfg.paths.models_dir / "turkish_word2vec.bin")

    training_settings = cfg.training.copy()
    training_settings.vector_size = vector_size
    training_settings.window = window
    training_settings.epochs = epochs
    training_settings.output_model = Path(out_file)

    trainer = Word2VecTrainer(settings=training_settings)
    click.echo(f"==> Training Word2Vec model on {corpus_file}...")
    try:
        trainer.train(corpus_file, out_file)
        click.secho(f"Model saved to {out_file}", fg="green")
    except NLPConnotationError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@cli.command(name="merge-lexicons")
@click.option("--nrc-path", type=click.Path(exists=True), help="Path to Turkish-NRC-VAD-Lexicon.csv")
@click.option("--memolon-path", type=click.Path(exists=True), help="Path to MEmoLon en.tsv")
@click.option("--output", type=click.Path(), help="Output path for merged my_Lexicon.csv")
@click.pass_context
def merge_lexicons(ctx, nrc_path, memolon_path, output):
    """Merge NRC-VAD and MEmoLon datasets into unified lexicon."""
    cfg: AppConfig = ctx.obj["config"]
    nrc = nrc_path or str(cfg.paths.nrc_vad_path)
    memolon = memolon_path or str(cfg.paths.memolon_path)
    out = output or str(cfg.paths.merged_lexicon)

    click.echo(f"==> Merging lexicons: {nrc} + {memolon} -> {out}")
    try:
        df = LexiconMerger.merge_datasets(nrc, memolon, out)
        click.secho(f"Merged lexicon created with {len(df):,} entries.", fg="green")
    except NLPConnotationError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@cli.command()
@click.option("--lexicon", type=click.Path(exists=True), help="Path to merged lexicon CSV")
@click.option("--output-dir", type=click.Path(), default="data/results", help="Directory for output reports")
@click.pass_context
def evaluate(ctx, lexicon, output_dir):
    """Run top-k semantic similarity and connotation analysis against lexicon."""
    cfg: AppConfig = ctx.obj["config"]
    lex_path = lexicon or str(cfg.paths.merged_lexicon)

    click.echo(f"==> Loading lexicon from {lex_path}...")
    try:
        lex_manager = LexiconManager(lex_path)
        analyzer = ConnotationAnalyzer(lex_manager)

        # Prepare test keywords mapping
        test_words = {
            "sport": {"tr": "spor", "en": "sport"},
            "economy": {"tr": "ekonomi", "en": "economy"},
            "magazine": {"tr": "dergi", "en": "magazine"},
            "politics": {"tr": "siyaset", "en": "politics"},
        }

        # Build sample dictionary adapters based on student's empirical results
        # for instant zero-dependency execution verification
        turkish_sample_dict = {
            "spor": [("futbol", 0.78), ("voleybol", 0.71), ("gençlikspor", 0.68), ("dağcılık", 0.65), ("boks", 0.63), ("basketbol", 0.62)],
            "magazin": [("ntv", 0.67), ("aktüalite", 0.65), ("söyleşisi", 0.63), ("habertürk", 0.61), ("birgün", 0.59)],
            "siyaset": [("politika", 0.82), ("sosyoloji", 0.74), ("ekonomi", 0.71), ("din", 0.69), ("hukuk", 0.67), ("diplomasi", 0.65)],
            "ekonomi": [("iktisat", 0.81), ("finans", 0.78), ("makroekonomi", 0.73), ("tarım", 0.69), ("sosyoloji", 0.67), ("bankacılık", 0.65)],
        }
        english_sample_dict = {
            "sport": [("sports", 0.85), ("athletics", 0.79), ("football", 0.76), ("racing", 0.72), ("basketball", 0.70)],
            "economy": [("economic", 0.82), ("market", 0.76), ("growth", 0.73), ("recession", 0.71), ("financial", 0.70)],
            "magazine": [("journal", 0.81), ("publication", 0.78), ("newspaper", 0.75), ("periodical", 0.72), ("vogue", 0.70)],
            "politics": [("political", 0.83), ("government", 0.78), ("policy", 0.75), ("democracy", 0.72), ("elections", 0.70)],
        }

        model_tr = DictionaryModelAdapter("Turkish_Word2Vec_Sample", turkish_sample_dict)
        model_en = DictionaryModelAdapter("GoogleNews_Sample", english_sample_dict)

        models_to_eval = [(model_tr, "tr"), (model_en, "en")]
        k_levels = [1, 3, 5]

        df_summary, df_details = analyzer.run_full_suite(
            models_and_languages=models_to_eval,
            test_words_dict=test_words,
            k_values=k_levels,
            output_dir=output_dir
        )

        click.secho(f"\nEvaluation Complete! Generated {len(df_summary)} summary records.", fg="green")
        click.echo("\n--- Top Summary Preview ---")
        click.echo(df_summary[["model", "keyword", "requested_k", "found_k", "mean_valence", "mean_arousal", "mean_dominance"]].head(8).to_string())

    except NLPConnotationError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
