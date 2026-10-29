"""
Unit tests for TurkishTokenizer.
"""

import unittest
from nlp_connotation.data.tokenizer import TurkishTokenizer, tokenize_turkish


class TestTurkishTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = TurkishTokenizer(min_token_len=2, max_token_len=50, lowercase=True)

    def test_turkish_lowercasing(self):
        # Crucial Turkish casing: 'I' -> 'ı' and 'İ' -> 'i'
        self.assertEqual(self.tokenizer.to_turkish_lower("IŞIK"), "ışık")
        self.assertEqual(self.tokenizer.to_turkish_lower("İSTANBUL"), "istanbul")
        self.assertEqual(self.tokenizer.to_turkish_lower("ÇIĞIR"), "çığır")
        self.assertEqual(self.tokenizer.to_turkish_lower("ÖĞRETMEN"), "öğretmen")
        self.assertEqual(self.tokenizer.to_turkish_lower("ŞÜPHE"), "şüphe")

    def test_token_splitting_and_filtering(self):
        sample = "Doğal Dil İşleme (NLP), yapay zekanın çok önemli bir dalıdır! _test_ a"
        tokens = self.tokenizer.tokenize(sample)

        # 'NLP' -> 'nlp'
        self.assertIn("nlp", tokens)
        self.assertIn("doğal", tokens)
        self.assertIn("dil", tokens)
        self.assertIn("işleme", tokens)
        self.assertIn("yapay", tokens)
        self.assertIn("zekanın", tokens)

        # Tokens shorter than min_token_len=2 ('a') or starting with '_' ('_test_') must be discarded
        self.assertNotIn("a", tokens)
        self.assertNotIn("_test_", tokens)

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])
        self.assertEqual(self.tokenizer.to_turkish_lower(""), "")


if __name__ == "__main__":
    unittest.main()
