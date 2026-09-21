"""
Unit Tests for Text Preprocessing Module.
"""

import pandas as pd
import pytest
from src.preprocessing import PreprocessingConfig, TextCleaner, build_tfidf_vectorizer, preprocess_series


def test_text_cleaner_urls():
    cleaner = TextCleaner()
    text = "Wildfire in Texas! See details https://t.co/xyz123 and http://example.com/fire"
    cleaned = cleaner.clean_text(text)
    assert "https" not in cleaned
    assert "http" not in cleaned
    assert "wildfire in texas" in cleaned


def test_text_cleaner_mentions_and_html_entities():
    cleaner = TextCleaner()
    text = "Alert @RedCross &amp; @FEMA! Heavy rain &lt; storm &gt; expected!"
    cleaned = cleaner.clean_text(text)
    assert "@RedCross" not in cleaned
    assert "@FEMA" not in cleaned
    assert "&amp;" not in cleaned
    assert "&lt;" not in cleaned
    assert "alert" in cleaned
    assert "storm" in cleaned


def test_text_cleaner_hashtags():
    cleaner = TextCleaner()
    text = "Emergency evacuation ordered #EarthquakeAlert #California"
    cleaned = cleaner.clean_text(text)
    assert "#" not in cleaned
    assert "earthquakealert" in cleaned
    assert "california" in cleaned


def test_text_cleaner_empty_and_whitespace():
    cleaner = TextCleaner()
    assert cleaner.clean_text("") == ""
    assert cleaner.clean_text("   \n\t  ") == ""
    assert cleaner.clean_text(None) == ""


def test_preprocess_series():
    s = pd.Series(["Hello @world!", "Check http://link.com #test"])
    cleaned_s = preprocess_series(s)
    assert cleaned_s.iloc[0] == "hello !"
    assert cleaned_s.iloc[1] == "check test"


def test_build_tfidf_vectorizer():
    vec = build_tfidf_vectorizer(max_features=500, ngram_range=(1, 2))
    assert vec.max_features == 500
    assert vec.ngram_range == (1, 2)
    X = vec.fit_transform(["building fire in downtown", "everyday sunny morning walk"])
    assert X.shape[0] == 2
