"""
Unit tests for configuration loading and validation.
"""

import unittest
from pathlib import Path
from nlp_connotation.core.config import AppConfig


class TestConfig(unittest.TestCase):

    def test_load_default_yaml_config(self):
        cfg_path = Path("configs/default_config.yaml")
        if cfg_path.exists():
            cfg = AppConfig.from_yaml(cfg_path)
            self.assertEqual(cfg.project.name, "turkish-word2vec-connotation")
            self.assertEqual(cfg.training.vector_size, 300)
            self.assertIn("sport", cfg.evaluation.test_words)

    def test_default_config_instantiation(self):
        cfg = AppConfig()
        self.assertEqual(cfg.tokenization.min_token_len, 2)
        self.assertTrue(cfg.tokenization.lowercase)


if __name__ == "__main__":
    unittest.main()
