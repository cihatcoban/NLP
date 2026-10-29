"""
Unit tests for ConnotationAnalyzer.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from nlp_connotation.analysis.connotation import ConnotationAnalyzer
from nlp_connotation.lexicons.manager import LexiconManager
from nlp_connotation.models.registry import DictionaryModelAdapter


class TestConnotationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create small test lexicon
        self.lex_path = self.dir_path / "test_lex.csv"
        df = pd.DataFrame({
            "English Word": ["football", "basketball"],
            "Turkish Word": ["futbol", "basketbol"],
            "Valence": [0.70, 0.80],
            "Arousal": [0.60, 0.50],
            "Dominance": [0.65, 0.75],
            "Joy": [3.0, 4.0],
            "Anger": [1.0, 1.0],
            "Sadness": [1.0, 1.0],
            "Fear": [1.0, 1.0],
            "Disgust": [1.0, 1.0],
        })
        df.to_csv(self.lex_path, index=False)
        self.lex_manager = LexiconManager(self.lex_path)
        self.analyzer = ConnotationAnalyzer(self.lex_manager)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_word_handling_and_mean_calculation(self):
        # futbol is in lexicon, voleybol is absent!
        mock_neighbors = {
            "spor": [("futbol", 0.9), ("voleybol", 0.8), ("basketbol", 0.7)]
        }
        model = DictionaryModelAdapter("TestModel", mock_neighbors)

        # Request top-3
        res = self.analyzer.evaluate_top_k(model, "spor", k=3, language="tr")

        self.assertEqual(res.requested_k, 3)
        # voleybol is absent, so found_k must be 2!
        self.assertEqual(res.found_k, 2)
        self.assertEqual(len(res.neighbors), 3)

        # Means must be average of futbol (0.70) and basketbol (0.80) -> 0.75
        self.assertAlmostEqual(res.mean_metrics.valence, 0.75)
        self.assertAlmostEqual(res.mean_metrics.arousal, 0.55)
        self.assertAlmostEqual(res.mean_metrics.joy, 3.5)


if __name__ == "__main__":
    unittest.main()
