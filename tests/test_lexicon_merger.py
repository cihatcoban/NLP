"""
Unit tests for LexiconMerger and LexiconManager.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from nlp_connotation.lexicons.merger import LexiconMerger
from nlp_connotation.lexicons.manager import LexiconManager
from nlp_connotation.core.exceptions import LexiconError


class TestLexiconModules(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create mock NRC-VAD csv
        self.nrc_path = self.dir_path / "nrc.csv"
        df_nrc = pd.DataFrame({
            "English Word": ["happy", "fearful", "unknown_word"],
            "Valence": [0.95, 0.20, 0.50],
            "Arousal": [0.80, 0.85, 0.50],
            "Dominance": [0.85, 0.25, 0.50],
            "Turkish Word": ["mutlu", "korkulu", "bilinmeyen"],
        })
        df_nrc.to_csv(self.nrc_path, sep=";", index=False)

        # Create mock MEmoLon tsv
        self.memolon_path = self.dir_path / "memolon.tsv"
        df_memolon = pd.DataFrame({
            "word": ["happy", "fearful"],
            "joy": [4.5, 1.1],
            "anger": [1.0, 2.0],
            "sadness": [1.0, 3.5],
            "fear": [1.0, 4.8],
            "disgust": [1.0, 1.5],
        })
        df_memolon.to_csv(self.memolon_path, sep="\t", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_vectorized_merge(self):
        out_csv = self.dir_path / "merged.csv"
        merged_df = LexiconMerger.merge_datasets(
            self.nrc_path,
            self.memolon_path,
            output_csv_path=out_csv,
            sep_nrc=";",
            sep_memolon="\t"
        )

        self.assertEqual(len(merged_df), 3)
        self.assertIn("Joy", merged_df.columns)
        self.assertIn("Fear", merged_df.columns)
        
        # Verify join correctness
        happy_row = merged_df[merged_df["English Word"] == "happy"].iloc[0]
        self.assertAlmostEqual(happy_row["Valence"], 0.95)
        self.assertAlmostEqual(happy_row["Joy"], 4.5)

    def test_lexicon_manager_lookups(self):
        out_csv = self.dir_path / "merged.csv"
        LexiconMerger.merge_datasets(self.nrc_path, self.memolon_path, output_csv_path=out_csv)

        manager = LexiconManager(out_csv)
        self.assertTrue(manager.contains("mutlu", lang="tr"))
        self.assertTrue(manager.contains("happy", lang="en"))
        self.assertFalse(manager.contains("nonexistent", lang="tr"))

        metrics_tr = manager.get_metrics("mutlu", lang="tr")
        self.assertIsNotNone(metrics_tr)
        self.assertAlmostEqual(metrics_tr.valence, 0.95)
        self.assertAlmostEqual(metrics_tr.joy, 4.5)


if __name__ == "__main__":
    unittest.main()
