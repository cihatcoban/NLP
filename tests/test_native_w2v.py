"""
Unit tests for Pure-NumPy NativeWord2VecModel and NativeKeyedVectors.
"""

import unittest
import tempfile
from pathlib import Path

from nlp_connotation.models.native_w2v import NativeWord2VecModel, NativeKeyedVectors
from nlp_connotation.models.registry import ModelRegistry


class TestNativeWord2Vec(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.sentences = [
            ["futbol", "voleybol", "basketbol", "spor", "olimpiyat", "stadyum"],
            ["futbol", "spor", "kaleci", "gol", "stadyum", "şampiyon"],
            ["ekonomi", "finans", "banka", "faiz", "borsa", "iktisat"],
            ["ekonomi", "iktisat", "enflasyon", "yatırım", "finans", "banka"],
            ["siyaset", "politika", "meclis", "seçim", "demokrasi", "parti"],
        ] * 5  # Repeat so min_count=2 is satisfied

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_and_save_binary_format(self):
        model = NativeWord2VecModel(
            sentences=self.sentences,
            vector_size=32,
            window=3,
            min_count=2,
            epochs=5,
        )
        self.assertIn("spor", model.wv)
        self.assertIn("ekonomi", model.wv)

        bin_path = self.dir_path / "test_w2v.bin"
        model.wv.save_word2vec_format(bin_path, binary=True)
        self.assertTrue(bin_path.exists())

        # Reload via ModelRegistry (NativeKeyedVectors fallback)
        registry = ModelRegistry()
        adapter = registry.load_local_word2vec_format("test_bin", bin_path, binary=True)
        self.assertTrue(adapter.contains("spor"))

        neighbors = adapter.most_similar("spor", top_n=3)
        self.assertEqual(len(neighbors), 3)
        self.assertNotEqual(neighbors[0][0], "spor")

    def test_save_and_load_text_format(self):
        model = NativeWord2VecModel(
            sentences=self.sentences,
            vector_size=16,
            window=2,
            min_count=2,
            epochs=3,
        )
        txt_path = self.dir_path / "test_w2v.vec"
        model.wv.save_word2vec_format(txt_path, binary=False)

        loaded = NativeKeyedVectors.load_word2vec_format(txt_path, binary=False)
        self.assertEqual(len(loaded), len(model.wv))
        self.assertIn("siyaset", loaded)


if __name__ == "__main__":
    unittest.main()
