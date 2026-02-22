"""
Tests for the example articles component in streamlit_app.py
"""

import pytest
from streamlit_app import EXAMPLE_ARTICLES, load_example


def test_example_articles_dictionary_exists():
    """Test that EXAMPLE_ARTICLES dictionary is defined."""
    assert EXAMPLE_ARTICLES is not None
    assert isinstance(EXAMPLE_ARTICLES, dict)


def test_example_articles_has_minimum_examples():
    """Test that at least 2 example articles are provided."""
    assert len(EXAMPLE_ARTICLES) >= 2


def test_example_articles_has_credible_example():
    """Test that a credible example article exists."""
    assert "Example Credible Article" in EXAMPLE_ARTICLES


def test_example_articles_has_suspicious_example():
    """Test that a suspicious example article exists."""
    assert "Example Suspicious Article" in EXAMPLE_ARTICLES


def test_credible_article_has_content():
    """Test that the credible article has substantial content."""
    credible_article = EXAMPLE_ARTICLES["Example Credible Article"]
    assert len(credible_article) >= 50  # Meets minimum length requirement
    assert isinstance(credible_article, str)


def test_suspicious_article_has_content():
    """Test that the suspicious article has substantial content."""
    suspicious_article = EXAMPLE_ARTICLES["Example Suspicious Article"]
    assert len(suspicious_article) >= 50  # Meets minimum length requirement
    assert isinstance(suspicious_article, str)


def test_load_example_credible():
    """Test loading the credible example article."""
    article = load_example("Example Credible Article")
    assert article == EXAMPLE_ARTICLES["Example Credible Article"]
    assert len(article) > 0


def test_load_example_suspicious():
    """Test loading the suspicious example article."""
    article = load_example("Example Suspicious Article")
    assert article == EXAMPLE_ARTICLES["Example Suspicious Article"]
    assert len(article) > 0


def test_load_example_invalid_name():
    """Test that loading a non-existent example raises KeyError."""
    with pytest.raises(KeyError):
        load_example("Non-existent Article")


def test_credible_article_characteristics():
    """Test that credible article has expected characteristics."""
    credible_article = EXAMPLE_ARTICLES["Example Credible Article"]
    
    # Should contain credible indicators
    assert "University" in credible_article or "study" in credible_article
    assert "research" in credible_article or "Dr." in credible_article
    
    # Should not have excessive capitalization or sensationalism
    all_caps_words = [word for word in credible_article.split() if word.isupper() and len(word) > 3]
    assert len(all_caps_words) < 5  # Minimal all-caps words


def test_suspicious_article_characteristics():
    """Test that suspicious article has expected characteristics."""
    suspicious_article = EXAMPLE_ARTICLES["Example Suspicious Article"]
    
    # Should contain suspicious indicators
    # Check for sensational language patterns
    suspicious_patterns = ["SHOCKING", "EXPLOSIVE", "ADMIT", "HIDING", "SILENCED"]
    found_patterns = sum(1 for pattern in suspicious_patterns if pattern in suspicious_article)
    assert found_patterns >= 2  # Should have multiple sensational patterns
    
    # Should have excessive capitalization
    all_caps_words = [word for word in suspicious_article.split() if word.isupper() and len(word) > 3]
    assert len(all_caps_words) >= 3  # Multiple all-caps words


def test_articles_meet_validation_requirements():
    """Test that both example articles meet input validation requirements."""
    from streamlit_app import InputValidator
    
    for article_name, article_text in EXAMPLE_ARTICLES.items():
        is_valid, error_msg = InputValidator.validate(article_text)
        assert is_valid, f"{article_name} failed validation: {error_msg}"
        assert error_msg == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
