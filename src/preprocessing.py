"""
Text Preprocessing Module for Disaster Tweets Classification.

Provides robust cleaning for noisy social media text (handles, URLs, HTML entities,
hashtags, punctuation, and emojis) along with vectorization/tokenization helpers.
"""

import html
import re
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class PreprocessingConfig:
    """Configuration settings for tweet text preprocessing."""
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    clean_hashtags: bool = True
    remove_html_entities: bool = True
    normalize_whitespace: bool = True
    strip_punctuation: bool = False
    min_tweet_length: int = 2


class TextCleaner:
    """
    Robust text cleaner specifically designed for noisy social media posts and tweets.
    """

    # Compiled Regex Patterns for optimal performance
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z0-9#]+;")
    MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")
    NON_PRINTABLE_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def clean_text(self, text: str) -> str:
        """
        Cleans a single input tweet string according to config rules.

        Args:
            text (str): Raw tweet text string.

        Returns:
            str: Cleaned and normalized text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
        if self.config.remove_html_entities:
            text = html.unescape(text)
            text = self.HTML_ENTITY_PATTERN.sub(" ", text)

        # 2. Remove URLs
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub(" ", text)

        # 3. Remove user mentions (@user)
        if self.config.remove_mentions:
            text = self.MENTION_PATTERN.sub(" ", text)

        # 4. Clean hashtags (e.g., #earthquake -> earthquake)
        if self.config.clean_hashtags:
            text = self.HASHTAG_PATTERN.sub(r"\1", text)

        # 5. Lowercase if configured
        if self.config.lowercase:
            text = text.lower()

        # 6. Remove non-printable control characters
        text = self.NON_PRINTABLE_PATTERN.sub(" ", text)

        # 7. Normalize whitespaces
        if self.config.normalize_whitespace:
            text = self.MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Cleans a batch list of tweet strings.

        Args:
            texts (List[str]): List of raw tweet strings.

        Returns:
            List[str]: List of cleaned tweet strings.
        """
        return [self.clean_text(t) for t in texts]


def preprocess_series(
    series: pd.Series,
    config: Optional[PreprocessingConfig] = None
) -> pd.Series:
    """
    Applies TextCleaner to a Pandas Series of text.

    Args:
        series (pd.Series): Pandas Series containing text strings.
        config (Optional[PreprocessingConfig]): Preprocessing configuration.

    Returns:
        pd.Series: Pandas Series of cleaned strings.
    """
    cleaner = TextCleaner(config=config)
    return series.astype(str).apply(cleaner.clean_text)


def build_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1
) -> TfidfVectorizer:
    """
    Factory function to build a pre-configured TF-IDF Vectorizer.

    Args:
        max_features (int): Maximum vocabulary size.
        ngram_range (tuple): N-gram range tuple (min_n, max_n).
        min_df (int): Minimum document frequency (default=1).

    Returns:
        TfidfVectorizer: Unfitted TF-IDF Vectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode"
    )

