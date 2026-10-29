"""
Turkish-aware tokenizer and text normalizer.
Handles Turkish-specific character casing (I -> ı, İ -> i), Unicode normalization,
and token length constraints.
"""

import re
import unicodedata
from typing import List, Optional


class TurkishTokenizer:
    """
    Production-grade Turkish tokenizer.
    Handles distinct Turkish lowercase mapping and handles diacritics cleanly.
    """

    # Exact Unicode mapping for Turkish lowercasing
    TURKISH_LOWER_MAP = {
        ord('I'): 'ı',
        ord('İ'): 'i',
        ord('A'): 'a',
        ord('B'): 'b',
        ord('C'): 'c',
        ord('Ç'): 'ç',
        ord('D'): 'd',
        ord('E'): 'e',
        ord('F'): 'f',
        ord('G'): 'g',
        ord('Ğ'): 'ğ',
        ord('H'): 'h',
        ord('J'): 'j',
        ord('K'): 'k',
        ord('L'): 'l',
        ord('M'): 'm',
        ord('N'): 'n',
        ord('O'): 'o',
        ord('Ö'): 'ö',
        ord('P'): 'p',
        ord('R'): 'r',
        ord('S'): 's',
        ord('Ş'): 'ş',
        ord('T'): 't',
        ord('U'): 'u',
        ord('Ü'): 'ü',
        ord('V'): 'v',
        ord('Y'): 'y',
        ord('Z'): 'z',
    }

    # Regex for splitting tokens on whitespace, punctuation, and non-alphanumeric chars
    WORD_SPLIT_PATTERN = re.compile(r"[^\w]+", re.UNICODE)

    def __init__(
        self,
        min_token_len: int = 2,
        max_token_len: int = 50,
        lowercase: bool = True,
        strip_numbers: bool = False,
    ):
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.lowercase = lowercase
        self.strip_numbers = strip_numbers

    def to_turkish_lower(self, text: str) -> str:
        """
        Converts text to lowercase according to Turkish alphabet rules:
        'I' -> 'ı' and 'İ' -> 'i'.
        """
        if not text:
            return ""
        return text.translate(self.TURKISH_LOWER_MAP).lower()

    def tokenize(self, content: str) -> List[str]:
        """
        Tokenizes Turkish text into a list of normalized tokens.
        Applies NFC normalization, lowercasing, and boundary/length validation.
        """
        if not content:
            return []

        # Normalize unicode representation
        text = unicodedata.normalize("NFC", content)

        if self.lowercase:
            text = self.to_turkish_lower(text)

        # Split tokens
        raw_tokens = self.WORD_SPLIT_PATTERN.split(text)

        tokens: List[str] = []
        for token in raw_tokens:
            if not token or token.startswith("_"):
                continue

            token_len = len(token)
            if not (self.min_token_len <= token_len <= self.max_token_len):
                continue

            if self.strip_numbers and token.isnumeric():
                continue

            tokens.append(token)

        return tokens


def tokenize_turkish(
    content: str,
    min_len: int = 2,
    max_len: int = 50,
    lowercase: bool = True
) -> List[str]:
    """Convenience functional wrapper for TurkishTokenizer."""
    tokenizer = TurkishTokenizer(
        min_token_len=min_len,
        max_token_len=max_len,
        lowercase=lowercase
    )
    return tokenizer.tokenize(content)
